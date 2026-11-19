/* ============================================================================
   SPORTMONKS FOOTBALL API v3  ->  PostgreSQL ANALYSIS DATABASE
   ============================================================================

   Run with:   psql -v ON_ERROR_STOP=1 -d your_database -f schema.sql
   Target:     PostgreSQL 14+ (tested on 16). Safe to run on an empty database.

   DESIGN RULES
   ------------
   1. Common IDs everywhere. Every primary key is the Sportmonks ID, named
      <entity>_id (team_id, player_id, referee_id, fixture_id, type_id ...).
      The same name is used in every table that refers to it, so joins are
      always  JOIN ... USING (team_id)  or  ON x.player_id = y.player_id.
      Sportmonks calls a team "participant" in fixture data; here it is always
      team_id. Columns with two references to the same thing are prefixed
      (home_team_id / away_team_id, from_team_id / to_team_id,
      related_player_id, nationality_id -> countries).

   2. Every coded value is a type_id pointing to core.types (event types,
      statistic types, official roles, positions, lineup types, etc.).
      Filter by core.types.developer_name (e.g. 'YELLOWCARD', 'FOULS'),
      which is stable and readable.

   3. Large match-level tables carry league_id and season_id copied from
      the fixture (filled automatically by trigger), so most analysis can
      filter by league/season without joining fixtures.

   4. Statistic values arrive as JSON objects. The original object is kept in
      `data` and a numeric `value` column is derived automatically
      (core.jnum) so you can AVG/SUM immediately.

   5. Raw API responses can be kept in raw.api_responses so tables can be
      rebuilt without re-downloading.

   6. raw.relationships is the single map of every reference between tables.
      It creates the indexes on every reference column and the enforced
      foreign keys, and raw.missing_references() lists IDs you still need
      to fetch (e.g. players referenced by events but not yet downloaded).

   SCHEMAS
   -------
   raw       landing zone, sync bookkeeping, relationship map
   core      geography, types, states
   football  competitions, teams, people, fixtures and everything inside them,
             standings, season statistics, transfers, injuries, predictions
   odds      bookmakers, markets, pre-match / in-play / premium odds + history
   analysis  ready-to-query views (fixture overview, team & player match
             stats, referee discipline, referee season summary)
   ========================================================================= */

BEGIN;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS football;
CREATE SCHEMA IF NOT EXISTS odds;
CREATE SCHEMA IF NOT EXISTS analysis;


/* ============================================================================
   HELPER FUNCTIONS
   ========================================================================= */

-- Text -> numeric. Handles '1.85', '45.5%', ' 12 '. Returns NULL otherwise.
CREATE OR REPLACE FUNCTION core.num(t text) RETURNS numeric
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE s text := btrim(replace(t, '%', ''));
BEGIN
    IF s ~ '^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' THEN
        RETURN s::numeric;
    END IF;
    RETURN NULL;
END $$;

-- JSON -> headline number. Numbers and numeric strings are returned as-is.
-- Objects are searched for value / total / all / count / average in that
-- order, e.g. {"value": 12} -> 12, {"all": {"count": 50, ...}} -> 50.
CREATE OR REPLACE FUNCTION core.jnum(j jsonb) RETURNS numeric
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
    CASE jsonb_typeof(j)
        WHEN 'number'  THEN RETURN (j #>> '{}')::numeric;
        WHEN 'string'  THEN RETURN core.num(j #>> '{}');
        WHEN 'boolean' THEN RETURN CASE WHEN (j #>> '{}')::boolean THEN 1 ELSE 0 END;
        WHEN 'object'  THEN RETURN COALESCE(core.jnum(j -> 'value'), core.jnum(j -> 'total'),
                                            core.jnum(j -> 'all'),   core.jnum(j -> 'count'),
                                            core.jnum(j -> 'average'));
        ELSE RETURN NULL;
    END CASE;
END $$;


/* ============================================================================
   RAW / BOOKKEEPING
   ========================================================================= */

CREATE TABLE raw.api_responses (
    response_id  bigserial   PRIMARY KEY,
    endpoint     text        NOT NULL,            -- e.g. /football/fixtures/multi/1,2,3
    params       jsonb,                           -- includes, filters, page
    entity       text,                            -- 'fixture', 'team', 'season' ...
    entity_id    bigint,                          -- set when the response is one entity
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    payload      jsonb       NOT NULL
);
CREATE INDEX ix_api_responses_entity  ON raw.api_responses (entity, entity_id);
CREATE INDEX ix_api_responses_fetched ON raw.api_responses USING brin (fetched_at);

-- What has been synced, so loaders can resume / update incrementally.
CREATE TABLE raw.sync_state (
    entity          text        NOT NULL,         -- 'fixtures', 'odds', 'season_stats' ...
    scope_key       text        NOT NULL,         -- e.g. 'season:23744'
    last_synced_at  timestamptz,
    status          text,                         -- 'ok', 'partial', 'error'
    detail          jsonb,
    PRIMARY KEY (entity, scope_key)
);

-- Map of every reference between tables (filled at the end of this file).
CREATE TABLE raw.relationships (
    tbl       text    NOT NULL,     -- schema.table holding the reference
    col       text    NOT NULL,     -- referencing column
    ref_tbl   text    NOT NULL,     -- schema.table referenced
    ref_col   text    NOT NULL,     -- referenced primary key column
    enforced  boolean NOT NULL,     -- true = real FOREIGN KEY constraint
    PRIMARY KEY (tbl, col)
);


/* ============================================================================
   CORE: geography, types, states
   ========================================================================= */

CREATE TABLE core.continents (
    continent_id  bigint PRIMARY KEY,
    name          text,
    code          text
);

CREATE TABLE core.countries (
    country_id     bigint PRIMARY KEY,
    continent_id   bigint,
    name           text,
    official_name  text,
    fifa_name      text,
    iso2           text,
    iso3           text,
    latitude       numeric,
    longitude      numeric,
    borders        text[],
    image_path     text
);

CREATE TABLE core.regions (
    region_id   bigint PRIMARY KEY,
    country_id  bigint,
    name        text
);

CREATE TABLE core.cities (
    city_id     bigint PRIMARY KEY,
    country_id  bigint,
    region_id   bigint,
    name        text,
    latitude    numeric,
    longitude   numeric
);

-- Every coded value in Sportmonks: event types, statistic types, positions,
-- official roles, lineup types, score types, period types, transfer types ...
CREATE TABLE core.types (
    type_id         bigint PRIMARY KEY,
    name            text,
    code            text,
    developer_name  text,         -- stable identifier, e.g. YELLOWCARD, FOULS, REFEREE
    model_type      text,         -- 'event', 'statistic', 'position', 'referee' ...
    stat_group      text
);
CREATE INDEX ix_types_developer_name ON core.types (developer_name);
CREATE INDEX ix_types_model_type     ON core.types (model_type, developer_name);

-- Fixture states (NS, LIVE, HT, FT, AET, FT_PEN, POSTP, CANCL ...)
CREATE TABLE core.states (
    state_id        bigint PRIMARY KEY,
    state           text,
    name            text,
    short_name      text,
    developer_name  text
);


/* ============================================================================
   FOOTBALL: competition structure
   ========================================================================= */

CREATE TABLE football.leagues (
    league_id       bigint PRIMARY KEY,
    country_id      bigint,
    name            text,
    short_code      text,
    type            text,           -- league / cup ...
    sub_type        text,           -- domestic / international ...
    category        int,
    active          boolean,
    has_jerseys     boolean,
    last_played_at  timestamptz,
    image_path      text
);

CREATE TABLE football.seasons (
    season_id                  bigint PRIMARY KEY,
    league_id                  bigint NOT NULL,
    name                       text,           -- '2025/2026'
    is_current                 boolean,
    finished                   boolean,
    pending                    boolean,
    starting_at                date,
    ending_at                  date,
    tie_breaker_rule_id        bigint,
    standings_recalculated_at  timestamptz,
    games_in_current_week      boolean
);
CREATE INDEX ix_seasons_name ON football.seasons (league_id, name);

CREATE TABLE football.stages (
    stage_id             bigint PRIMARY KEY,
    league_id            bigint,
    season_id            bigint,
    type_id              bigint,
    name                 text,
    sort_order           int,
    finished             boolean,
    is_current           boolean,
    starting_at          date,
    ending_at            date,
    tie_breaker_rule_id  bigint
);

CREATE TABLE football.rounds (
    round_id     bigint PRIMARY KEY,
    league_id    bigint,
    season_id    bigint,
    stage_id     bigint,
    name         text,               -- usually the matchday number
    finished     boolean,
    is_current   boolean,
    starting_at  date,
    ending_at    date
);

CREATE TABLE football.groups (
    group_id     bigint PRIMARY KEY,
    league_id    bigint,
    season_id    bigint,
    stage_id     bigint,
    name         text,
    finished     boolean,
    is_current   boolean,
    pending      boolean,
    starting_at  date,
    ending_at    date
);

-- Two-legged ties
CREATE TABLE football.aggregates (
    aggregate_id    bigint PRIMARY KEY,
    league_id       bigint,
    season_id       bigint,
    stage_id        bigint,
    name            text,
    fixture_ids     bigint[],
    result          text,
    detail          text,
    winner_team_id  bigint
);


/* ============================================================================
   FOOTBALL: venues, teams, people
   ========================================================================= */

CREATE TABLE football.venues (
    venue_id       bigint PRIMARY KEY,
    country_id     bigint,
    city_id        bigint,
    name           text,
    address        text,
    zipcode        text,
    state          text,
    city_name      text,
    latitude       numeric,
    longitude      numeric,
    capacity       int,
    surface        text,
    national_team  boolean,
    image_path     text
);

CREATE TABLE football.teams (
    team_id         bigint PRIMARY KEY,
    country_id      bigint,
    venue_id        bigint,
    name            text,
    short_code      text,
    gender          text,
    type            text,            -- domestic / national
    founded         int,
    placeholder     boolean,
    last_played_at  timestamptz,
    image_path      text
);
CREATE INDEX ix_teams_name ON football.teams (name);

CREATE TABLE football.players (
    player_id             bigint PRIMARY KEY,
    country_id            bigint,
    nationality_id        bigint,     -- -> core.countries
    city_id               bigint,
    position_id           bigint,     -- -> core.types (positions are types)
    detailed_position_id  bigint,     -- -> core.types
    type_id               bigint,
    common_name           text,
    firstname             text,
    lastname              text,
    name                  text,
    display_name          text,
    height                int,        -- cm
    weight                int,        -- kg
    date_of_birth         date,
    gender                text,
    image_path            text
);
CREATE INDEX ix_players_name ON football.players (display_name);

CREATE TABLE football.coaches (
    coach_id        bigint PRIMARY KEY,
    player_id       bigint,           -- same person's player record, if any
    country_id      bigint,
    nationality_id  bigint,
    city_id         bigint,
    common_name     text,
    firstname       text,
    lastname        text,
    name            text,
    display_name    text,
    height          int,
    weight          int,
    date_of_birth   date,
    gender          text,
    image_path      text
);

CREATE TABLE football.referees (
    referee_id      bigint PRIMARY KEY,
    country_id      bigint,
    nationality_id  bigint,
    city_id         bigint,
    common_name     text,
    firstname       text,
    lastname        text,
    name            text,
    display_name    text,
    height          int,
    weight          int,
    date_of_birth   date,
    gender          text,
    image_path      text
);
CREATE INDEX ix_referees_name ON football.referees (display_name);

CREATE TABLE football.tv_stations (
    tv_station_id  bigint PRIMARY KEY,
    name           text,
    url            text,
    type           text,
    related_id     bigint,
    image_path     text
);

CREATE TABLE football.rivals (
    team_id        bigint NOT NULL,
    rival_team_id  bigint NOT NULL,
    PRIMARY KEY (team_id, rival_team_id)
);

CREATE TABLE football.trophies (
    trophy_id  bigint PRIMARY KEY,
    position   int,
    name       text
);

CREATE TABLE football.team_trophies (
    team_trophy_id  bigint PRIMARY KEY,   -- ParticipantTrophy.id
    team_id         bigint,
    league_id       bigint,
    season_id       bigint,
    trophy_id       bigint
);

-- FIFA / UEFA style rankings
CREATE TABLE football.rankings (
    ranking_id  bigint PRIMARY KEY,
    team_id     bigint,
    type        text,
    position    int,
    points      numeric
);


/* ============================================================================
   FOOTBALL: squads, coaching spells, transfers, injuries & suspensions
   ========================================================================= */

-- Season squads (season_id set) and player-team memberships (season_id NULL)
CREATE TABLE football.team_squads (
    squad_id              bigint PRIMARY KEY,
    team_id               bigint NOT NULL,
    player_id             bigint NOT NULL,
    season_id             bigint,
    transfer_id           bigint,
    position_id           bigint,
    detailed_position_id  bigint,
    jersey_number         int,
    captain               boolean,
    start_date            date,
    end_date              date
);
CREATE INDEX ix_team_squads_team_season ON football.team_squads (team_id, season_id);

CREATE TABLE football.coach_teams (
    coach_team_id  bigint PRIMARY KEY,
    coach_id       bigint NOT NULL,
    team_id        bigint NOT NULL,
    position_id    bigint,               -- head coach / assistant ...
    active         boolean,
    temporary      boolean,
    start_date     date,
    end_date       date
);

CREATE TABLE football.transfers (
    transfer_id           bigint PRIMARY KEY,
    player_id             bigint,
    type_id               bigint,        -- loan / transfer / free ...
    from_team_id          bigint,
    to_team_id            bigint,
    position_id           bigint,
    detailed_position_id  bigint,
    date                  date,
    career_ended          boolean,
    completed             boolean,
    completed_at          timestamptz,
    amount                numeric
);
CREATE INDEX ix_transfers_date ON football.transfers (date);

CREATE TABLE football.transfer_rumours (
    transfer_rumour_id  bigint PRIMARY KEY,
    player_id           bigint,
    position_id         bigint,
    from_team_id        bigint,
    to_team_id          bigint,
    transfer_fee_id     bigint,
    type_id             bigint,
    probability         text,
    source_name         text,
    source_url          text,
    amount              numeric,
    currency            text,
    date                date
);

-- Injuries and suspensions
CREATE TABLE football.sidelined (
    sidelined_id  bigint PRIMARY KEY,
    player_id     bigint,
    team_id       bigint,
    season_id     bigint,
    type_id       bigint,                -- injury / suspension type
    category      text,                  -- 'injury', 'suspension' ...
    start_date    date,
    end_date      date,
    games_missed  int,
    completed     boolean
);
CREATE INDEX ix_sidelined_dates ON football.sidelined (start_date, end_date);


/* ============================================================================
   FOOTBALL: fixtures (one row per match)
   ========================================================================= */

CREATE TABLE football.fixtures (
    fixture_id        bigint PRIMARY KEY,
    league_id         bigint NOT NULL,
    season_id         bigint NOT NULL,
    stage_id          bigint,
    round_id          bigint,
    group_id          bigint,
    aggregate_id      bigint,
    state_id          bigint,
    venue_id          bigint,
    home_team_id      bigint,    -- filled automatically from fixture_participants
    away_team_id      bigint,    -- filled automatically from fixture_participants
    name              text,      -- 'Bayern München vs Borussia Dortmund'
    starting_at       timestamptz,
    result_info       text,
    leg               text,
    details           text,
    length            int,       -- minutes
    placeholder       boolean,
    has_odds          boolean,
    has_premium_odds  boolean,
    ingested_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_fixtures_starting_at   ON football.fixtures (starting_at);
CREATE INDEX ix_fixtures_league_date   ON football.fixtures (league_id, starting_at);
CREATE INDEX ix_fixtures_season_round  ON football.fixtures (season_id, round_id);


/* ============================================================================
   FOOTBALL: everything inside a fixture
   (league_id / season_id on these tables are filled from the fixture)
   ========================================================================= */

-- Teams in the match
CREATE TABLE football.fixture_participants (
    fixture_id  bigint NOT NULL,
    team_id     bigint NOT NULL,
    location    text,                 -- 'home' / 'away'
    winner      boolean,
    position    int,                  -- league position at kickoff
    league_id   bigint,
    season_id   bigint,
    PRIMARY KEY (fixture_id, team_id)
);
CREATE INDEX ix_fixture_participants_team_season ON football.fixture_participants (team_id, season_id);

-- Scores by type (1ST_HALF, 2ND_HALF, CURRENT, ET, PENALTY_SHOOTOUT ...)
CREATE TABLE football.fixture_scores (
    score_id     bigint PRIMARY KEY,
    fixture_id   bigint NOT NULL,
    team_id      bigint,
    type_id      bigint,
    goals        int,
    location     text,                -- score.participant: 'home' / 'away'
    description  text,                -- 'CURRENT', '1ST_HALF' ...
    league_id    bigint,
    season_id    bigint
);
CREATE INDEX ix_fixture_scores_fixture_desc ON football.fixture_scores (fixture_id, description);

CREATE TABLE football.fixture_periods (
    period_id      bigint PRIMARY KEY,
    fixture_id     bigint NOT NULL,
    type_id        bigint,
    started        bigint,            -- unix timestamp
    ended          bigint,
    counts_from    int,
    ticking        boolean,
    sort_order     int,
    description    text,
    time_added     int,               -- stoppage time announced
    period_length  int,
    minutes        int,
    seconds        int,
    has_timer      boolean,
    league_id      bigint,
    season_id      bigint
);

-- Goals, cards, substitutions, VAR, penalties, delays (events + timeline)
CREATE TABLE football.fixture_events (
    event_id             bigint PRIMARY KEY,
    fixture_id           bigint NOT NULL,
    period_id            bigint,
    detailed_period_id   bigint,
    team_id              bigint,
    type_id              bigint,
    sub_type_id          bigint,
    section              text,
    player_id            bigint,
    related_player_id    bigint,      -- assist provider, player subbed off ...
    coach_id             bigint,      -- set when a coach is booked
    player_name          text,
    related_player_name  text,
    result               text,        -- score after a goal, e.g. '2-1'
    info                 text,
    addition             text,
    minute               int,
    extra_minute         int,
    injured              boolean,
    on_bench             boolean,
    rescinded            boolean,     -- card / goal later cancelled
    sort_order           int,
    league_id            bigint,
    season_id            bigint
);
CREATE INDEX ix_fixture_events_type_season ON football.fixture_events (type_id, season_id);
CREATE INDEX ix_fixture_events_fixture_min ON football.fixture_events (fixture_id, minute, extra_minute);

-- Starting XI and bench
CREATE TABLE football.fixture_lineups (
    lineup_id             bigint PRIMARY KEY,
    fixture_id            bigint NOT NULL,
    team_id               bigint,
    player_id             bigint,
    type_id               bigint,      -- LINEUP (starter) / BENCH
    position_id           bigint,
    detailed_position_id  bigint,
    formation_field       text,        -- grid position, e.g. '2:3'
    formation_position    int,
    jersey_number         int,
    player_name           text,
    league_id             bigint,
    season_id             bigint
);
CREATE INDEX ix_fixture_lineups_player_season ON football.fixture_lineups (player_id, season_id);

-- Player match statistics (minutes, rating, passes, tackles, duels ...)
CREATE TABLE football.fixture_lineup_details (
    lineup_detail_id  bigint PRIMARY KEY,
    lineup_id         bigint,
    fixture_id        bigint NOT NULL,
    team_id           bigint,
    player_id         bigint,
    type_id           bigint NOT NULL,
    data              jsonb,
    value             numeric GENERATED ALWAYS AS (core.jnum(data)) STORED,
    league_id         bigint,
    season_id         bigint
);
CREATE INDEX ix_lineup_details_player_type ON football.fixture_lineup_details (player_id, type_id);
CREATE INDEX ix_lineup_details_type_season ON football.fixture_lineup_details (type_id, season_id);

-- Team match statistics (shots, corners, fouls, possession ...)
CREATE TABLE football.fixture_statistics (
    statistic_id  bigint PRIMARY KEY,
    fixture_id    bigint NOT NULL,
    team_id       bigint,
    type_id       bigint NOT NULL,
    location      text,
    data          jsonb,
    value         numeric GENERATED ALWAYS AS (core.jnum(data)) STORED,
    league_id     bigint,
    season_id     bigint
);
CREATE UNIQUE INDEX ux_fixture_statistics ON football.fixture_statistics (fixture_id, team_id, type_id);
CREATE INDEX ix_fixture_statistics_type_season ON football.fixture_statistics (type_id, season_id);

CREATE TABLE football.fixture_formations (
    formation_id  bigint PRIMARY KEY,
    fixture_id    bigint NOT NULL,
    team_id       bigint,
    formation     text,               -- '4-2-3-1'
    location      text,
    league_id     bigint,
    season_id     bigint
);

-- Match officials: referee, assistants, fourth official, VAR, AVAR
CREATE TABLE football.fixture_officials (
    fixture_official_id  bigint PRIMARY KEY,
    fixture_id           bigint NOT NULL,
    referee_id           bigint NOT NULL,
    type_id              bigint,      -- role
    league_id            bigint,
    season_id            bigint
);
CREATE UNIQUE INDEX ux_fixture_officials ON football.fixture_officials (fixture_id, referee_id, type_id);
CREATE INDEX ix_fixture_officials_referee_season ON football.fixture_officials (referee_id, season_id);

CREATE TABLE football.fixture_coaches (
    fixture_id  bigint NOT NULL,
    coach_id    bigint NOT NULL,
    team_id     bigint,
    league_id   bigint,
    season_id   bigint,
    PRIMARY KEY (fixture_id, coach_id)
);

CREATE TABLE football.fixture_tv_stations (
    fixture_tv_station_id  bigint PRIMARY KEY,
    fixture_id             bigint NOT NULL,
    tv_station_id          bigint,
    country_id             bigint
);

-- Players missing a specific match (links to football.sidelined)
CREATE TABLE football.fixture_sidelined (
    fixture_sidelined_id  bigint PRIMARY KEY,
    fixture_id            bigint NOT NULL,
    sidelined_id          bigint,
    team_id               bigint,
    league_id             bigint,
    season_id             bigint
);

CREATE TABLE football.fixture_weather (
    weather_id          bigint PRIMARY KEY,
    fixture_id          bigint NOT NULL UNIQUE,
    venue_id            bigint,
    temperature_day     numeric,
    temperature_morning numeric,
    temperature_evening numeric,
    temperature_night   numeric,
    feels_like_day      numeric,
    wind_speed          numeric,
    wind_direction      numeric,
    humidity            numeric,      -- %
    pressure            numeric,
    clouds              numeric,      -- %
    description         text,
    icon                text,
    type                text,         -- forecast / current
    metric              text,         -- celcius / fahrenheit
    data                jsonb         -- full original object
);

-- Momentum trends per minute
CREATE TABLE football.fixture_trends (
    trend_id    bigint PRIMARY KEY,
    fixture_id  bigint NOT NULL,
    team_id     bigint,
    type_id     bigint,
    period_id   bigint,
    minute      int,
    value       numeric,
    league_id   bigint,
    season_id   bigint
);
CREATE INDEX ix_fixture_trends_fixture_type ON football.fixture_trends (fixture_id, type_id, minute);

-- Pressure Index per minute
CREATE TABLE football.fixture_pressure (
    pressure_id  bigint PRIMARY KEY,
    fixture_id   bigint NOT NULL,
    team_id      bigint,
    minute       int,
    pressure     numeric,
    league_id    bigint,
    season_id    bigint
);
CREATE INDEX ix_fixture_pressure_fixture_min ON football.fixture_pressure (fixture_id, minute);

-- Ball position samples (very large: ~thousands of rows per match)
CREATE TABLE football.fixture_ball_coordinates (
    ball_coordinate_id  bigint PRIMARY KEY,
    fixture_id          bigint NOT NULL,
    period_id           bigint,
    timer               text,
    x                   numeric,
    y                   numeric
);

-- Team-level expected goals and other expected metrics
CREATE TABLE football.fixture_xg (
    xg_id       bigint PRIMARY KEY,
    fixture_id  bigint NOT NULL,
    team_id     bigint,
    type_id     bigint NOT NULL,      -- EXPECTED_GOALS, EXPECTED_GOALS_ON_TARGET ...
    location    text,
    data        jsonb,
    value       numeric GENERATED ALWAYS AS (core.jnum(data)) STORED,
    league_id   bigint,
    season_id   bigint
);
CREATE INDEX ix_fixture_xg_type_season ON football.fixture_xg (type_id, season_id);

-- Player-level expected metrics
CREATE TABLE football.fixture_xg_players (
    xg_player_id  bigint PRIMARY KEY,
    fixture_id    bigint NOT NULL,
    team_id       bigint,
    player_id     bigint,
    lineup_id     bigint,
    type_id       bigint NOT NULL,
    data          jsonb,
    value         numeric GENERATED ALWAYS AS (core.jnum(data)) STORED,
    league_id     bigint,
    season_id     bigint
);
CREATE INDEX ix_fixture_xg_players_player_type ON football.fixture_xg_players (player_id, type_id);

-- Predicted lineups published before kickoff
CREATE TABLE football.fixture_expected_lineups (
    expected_lineup_id  bigint PRIMARY KEY,
    fixture_id          bigint NOT NULL,
    team_id             bigint,
    player_id           bigint,
    type_id             bigint,
    position_id         bigint,
    formation_field     text,
    formation_position  int,
    jersey_number       int,
    data                jsonb,
    league_id           bigint,
    season_id           bigint
);

-- Text commentary (Sportmonks commentary ids are strings)
CREATE TABLE football.fixture_commentaries (
    commentary_id      text   PRIMARY KEY,
    fixture_id         bigint NOT NULL,
    player_id          bigint,
    related_player_id  bigint,
    comment            text,
    minute             int,
    extra_minute       int,
    is_goal            boolean,
    is_important       boolean,
    sort_order         int
);

-- Match facts (structure varies, kept as JSON)
CREATE TABLE football.fixture_match_facts (
    match_fact_id  bigint PRIMARY KEY,
    fixture_id     bigint NOT NULL,
    team_id        bigint,
    type_id        bigint,
    data           jsonb
);

-- Pre-match and post-match news articles
CREATE TABLE football.fixture_news (
    news_id     bigint PRIMARY KEY,
    fixture_id  bigint,
    league_id   bigint,
    title       text,
    type        text                 -- 'prematch' / 'postmatch'
);

CREATE TABLE football.fixture_news_lines (
    news_line_id  bigint PRIMARY KEY,
    news_id       bigint NOT NULL,
    text          text,
    type          text
);

-- Metadata attached to any entity (fixture, team, league ...)
CREATE TABLE football.metadata (
    metadata_id  bigint PRIMARY KEY,
    entity_type  text   NOT NULL,     -- metadatable_type
    entity_id    bigint NOT NULL,     -- metadatable_id
    type_id      bigint,
    value_type   text,
    value        jsonb
);
CREATE INDEX ix_metadata_entity ON football.metadata (entity_type, entity_id);


/* ============================================================================
   FOOTBALL: standings and top scorers
   ========================================================================= */

CREATE TABLE football.standings (
    standing_id       bigint PRIMARY KEY,
    league_id         bigint,
    season_id         bigint NOT NULL,
    stage_id          bigint,
    group_id          bigint,
    round_id          bigint,
    team_id           bigint NOT NULL,
    standing_rule_id  bigint,
    position          int,
    result            text,           -- 'up' / 'down' / 'equal'
    points            int,
    is_live           boolean DEFAULT false
);
CREATE INDEX ix_standings_season_round ON football.standings (season_id, round_id, position);

-- Played, won, drawn, lost, goals for/against, home/away splits ...
CREATE TABLE football.standing_details (
    standing_detail_id  bigint PRIMARY KEY,
    standing_id         bigint NOT NULL,
    team_id             bigint,
    type_id             bigint NOT NULL,
    value               numeric
);

CREATE TABLE football.standing_form (
    standing_form_id  bigint PRIMARY KEY,
    standing_id       bigint NOT NULL,
    team_id           bigint,
    fixture_id        bigint,
    form              text,           -- 'W' / 'D' / 'L'
    sort_order        int
);

CREATE TABLE football.topscorers (
    topscorer_id  bigint PRIMARY KEY,
    season_id     bigint NOT NULL,
    stage_id      bigint,
    player_id     bigint,
    team_id       bigint,
    type_id       bigint,             -- goals / assists / cards
    position      int,
    total         int
);


/* ============================================================================
   FOOTBALL: season-level statistics
   (each parent row = one entity in one season; details = one row per type)
   ========================================================================= */

CREATE TABLE football.season_statistics (
    season_statistic_id  bigint PRIMARY KEY,
    season_id            bigint NOT NULL,      -- model_id
    type_id              bigint NOT NULL,
    relation_type        text,
    relation_id          bigint,
    data                 jsonb,
    value                numeric GENERATED ALWAYS AS (core.jnum(data)) STORED
);

CREATE TABLE football.team_season_stats (
    team_statistic_id  bigint PRIMARY KEY,
    team_id            bigint NOT NULL,
    season_id          bigint NOT NULL,
    has_values         boolean
);
CREATE UNIQUE INDEX ux_team_season_stats ON football.team_season_stats (team_id, season_id);

CREATE TABLE football.team_season_stat_details (
    team_statistic_detail_id  bigint PRIMARY KEY,
    team_statistic_id         bigint NOT NULL,
    type_id                   bigint NOT NULL,
    data                      jsonb,
    value                     numeric GENERATED ALWAYS AS (core.jnum(data)) STORED
);

CREATE TABLE football.player_season_stats (
    player_statistic_id  bigint PRIMARY KEY,
    player_id            bigint NOT NULL,
    team_id              bigint,
    season_id            bigint NOT NULL,
    position_id          bigint,
    jersey_number        int,
    has_values           boolean
);
CREATE INDEX ix_player_season_stats_season ON football.player_season_stats (season_id, team_id);

CREATE TABLE football.player_season_stat_details (
    player_statistic_detail_id  bigint PRIMARY KEY,
    player_statistic_id         bigint NOT NULL,
    type_id                     bigint NOT NULL,
    data                        jsonb,
    value                       numeric GENERATED ALWAYS AS (core.jnum(data)) STORED
);

CREATE TABLE football.coach_season_stats (
    coach_statistic_id  bigint PRIMARY KEY,
    coach_id            bigint NOT NULL,
    team_id             bigint,
    season_id           bigint NOT NULL
);

CREATE TABLE football.coach_season_stat_details (
    coach_statistic_detail_id  bigint PRIMARY KEY,
    coach_statistic_id         bigint NOT NULL,
    type_id                    bigint NOT NULL,
    data                       jsonb,
    value                      numeric GENERATED ALWAYS AS (core.jnum(data)) STORED
);

CREATE TABLE football.referee_season_stats (
    referee_statistic_id  bigint PRIMARY KEY,
    referee_id            bigint NOT NULL,
    season_id             bigint NOT NULL
);
CREATE UNIQUE INDEX ux_referee_season_stats ON football.referee_season_stats (referee_id, season_id);

CREATE TABLE football.referee_season_stat_details (
    referee_statistic_detail_id  bigint PRIMARY KEY,
    referee_statistic_id         bigint NOT NULL,
    type_id                      bigint NOT NULL,
    data                         jsonb,
    value                        numeric GENERATED ALWAYS AS (core.jnum(data)) STORED
);


/* ============================================================================
   FOOTBALL: predictions
   ========================================================================= */

CREATE TABLE football.predictions (
    prediction_id  bigint PRIMARY KEY,
    fixture_id     bigint NOT NULL,
    type_id        bigint NOT NULL,       -- market predicted, or value bet
    predictions    jsonb,                 -- e.g. {"home": 45.2, "draw": 27.1, "away": 27.7}
    league_id      bigint,
    season_id      bigint
);
CREATE INDEX ix_predictions_fixture_type ON football.predictions (fixture_id, type_id);

CREATE TABLE football.live_predictions (
    live_prediction_id  bigint PRIMARY KEY,
    fixture_id          bigint NOT NULL,
    period_id           bigint,
    type_id             bigint,
    minute              int,
    predictions         jsonb,
    league_id           bigint,
    season_id           bigint
);

-- How predictable a league is per market
CREATE TABLE football.league_probabilities (
    probability_id  bigint PRIMARY KEY,
    league_id       bigint NOT NULL,
    type_id         bigint,
    data            jsonb
);


/* ============================================================================
   ODDS
   ========================================================================= */

CREATE TABLE odds.bookmakers (
    bookmaker_id  bigint PRIMARY KEY,
    legacy_id     bigint,
    name          text
);

CREATE TABLE odds.markets (
    market_id                 bigint PRIMARY KEY,
    legacy_id                 bigint,
    name                      text,
    developer_name            text,
    has_winning_calculations  boolean
);

-- Standard pre-match odds (latest price per selection)
CREATE TABLE odds.prematch_odds (
    odd_id                   bigint PRIMARY KEY,
    fixture_id               bigint NOT NULL,
    market_id                bigint,
    bookmaker_id             bigint,
    label                    text,          -- '1', 'X', '2', 'Over' ...
    original_label           text,
    name                     text,
    market_description       text,
    value                    text,
    decimal_odds             numeric GENERATED ALWAYS AS (core.num(value)) STORED,
    probability              text,
    probability_pct          numeric GENERATED ALWAYS AS (core.num(probability)) STORED,
    dp3                      text,
    fractional               text,
    american                 text,
    winning                  boolean,       -- settled result of the selection
    stopped                  boolean,
    total                    text,          -- line for over/under markets
    handicap                 text,
    participants             text,
    latest_bookmaker_update  timestamptz,
    league_id                bigint,
    season_id                bigint
);
CREATE INDEX ix_prematch_odds_fixture_market ON odds.prematch_odds (fixture_id, market_id, bookmaker_id);
CREATE INDEX ix_prematch_odds_market_season  ON odds.prematch_odds (market_id, season_id);

CREATE TABLE odds.inplay_odds (
    odd_id                   bigint PRIMARY KEY,
    external_id              bigint,
    fixture_id               bigint NOT NULL,
    market_id                bigint,
    bookmaker_id             bigint,
    label                    text,
    name                     text,
    market_description       text,
    value                    text,
    decimal_odds             numeric GENERATED ALWAYS AS (core.num(value)) STORED,
    probability              text,
    probability_pct          numeric GENERATED ALWAYS AS (core.num(probability)) STORED,
    dp3                      text,
    fractional               text,
    american                 text,
    suspended                boolean,
    stopped                  boolean,
    total                    text,
    handicap                 text,
    participants             text,
    latest_bookmaker_update  timestamptz,
    league_id                bigint,
    season_id                bigint
);
CREATE INDEX ix_inplay_odds_fixture_market ON odds.inplay_odds (fixture_id, market_id, bookmaker_id);

-- Premium odds feed (more bookmakers/markets) ...
CREATE TABLE odds.premium_odds (
    odd_id                   bigint PRIMARY KEY,
    fixture_id               bigint NOT NULL,
    market_id                bigint,
    bookmaker_id             bigint,
    label                    text,
    name                     text,
    market_description       text,
    value                    text,
    decimal_odds             numeric GENERATED ALWAYS AS (core.num(value)) STORED,
    probability              text,
    probability_pct          numeric GENERATED ALWAYS AS (core.num(probability)) STORED,
    dp3                      text,
    fractional               text,
    american                 text,
    stopped                  boolean,
    total                    text,
    handicap                 text,
    latest_bookmaker_update  timestamptz,
    league_id                bigint,
    season_id                bigint
);
CREATE INDEX ix_premium_odds_fixture_market ON odds.premium_odds (fixture_id, market_id, bookmaker_id);
CREATE INDEX ix_premium_odds_market_season  ON odds.premium_odds (market_id, season_id);

-- ... with full price history (opening -> closing line movement)
CREATE TABLE odds.premium_odds_history (
    history_id        bigint PRIMARY KEY,
    odd_id            bigint NOT NULL,
    value             text,
    decimal_odds      numeric GENERATED ALWAYS AS (core.num(value)) STORED,
    probability       text,
    probability_pct   numeric GENERATED ALWAYS AS (core.num(probability)) STORED,
    dp3               text,
    fractional        text,
    american          text,
    bookmaker_update  timestamptz
);
CREATE INDEX ix_premium_odds_history_odd_time ON odds.premium_odds_history (odd_id, bookmaker_update);


/* ============================================================================
   RELATIONSHIP MAP
   enforced = true  -> real FOREIGN KEY (deferrable: checked at COMMIT, so
                       insert order inside a transaction doesn't matter)
   enforced = false -> indexed but not enforced. Used for people, teams,
                       venues, geography and sub-structures that Sportmonks
                       references before you may have downloaded them.
                       raw.missing_references() lists any gaps.
   Every referencing column gets an index either way.
   ========================================================================= */

INSERT INTO raw.relationships (tbl, col, ref_tbl, ref_col, enforced) VALUES
-- core
('core.countries','continent_id','core.continents','continent_id',false),
('core.regions','country_id','core.countries','country_id',false),
('core.cities','country_id','core.countries','country_id',false),
('core.cities','region_id','core.regions','region_id',false),
-- competition structure
('football.leagues','country_id','core.countries','country_id',false),
('football.seasons','league_id','football.leagues','league_id',true),
('football.stages','league_id','football.leagues','league_id',true),
('football.stages','season_id','football.seasons','season_id',true),
('football.stages','type_id','core.types','type_id',false),
('football.rounds','league_id','football.leagues','league_id',true),
('football.rounds','season_id','football.seasons','season_id',true),
('football.rounds','stage_id','football.stages','stage_id',false),
('football.groups','league_id','football.leagues','league_id',true),
('football.groups','season_id','football.seasons','season_id',true),
('football.groups','stage_id','football.stages','stage_id',false),
('football.aggregates','league_id','football.leagues','league_id',true),
('football.aggregates','season_id','football.seasons','season_id',true),
('football.aggregates','stage_id','football.stages','stage_id',false),
('football.aggregates','winner_team_id','football.teams','team_id',false),
-- venues, teams, people
('football.venues','country_id','core.countries','country_id',false),
('football.venues','city_id','core.cities','city_id',false),
('football.teams','country_id','core.countries','country_id',false),
('football.teams','venue_id','football.venues','venue_id',false),
('football.players','country_id','core.countries','country_id',false),
('football.players','nationality_id','core.countries','country_id',false),
('football.players','city_id','core.cities','city_id',false),
('football.players','position_id','core.types','type_id',false),
('football.players','detailed_position_id','core.types','type_id',false),
('football.players','type_id','core.types','type_id',false),
('football.coaches','player_id','football.players','player_id',false),
('football.coaches','country_id','core.countries','country_id',false),
('football.coaches','nationality_id','core.countries','country_id',false),
('football.coaches','city_id','core.cities','city_id',false),
('football.referees','country_id','core.countries','country_id',false),
('football.referees','nationality_id','core.countries','country_id',false),
('football.referees','city_id','core.cities','city_id',false),
('football.rivals','rival_team_id','football.teams','team_id',false),
('football.team_trophies','team_id','football.teams','team_id',false),
('football.team_trophies','league_id','football.leagues','league_id',false),
('football.team_trophies','season_id','football.seasons','season_id',false),
('football.team_trophies','trophy_id','football.trophies','trophy_id',false),
('football.rankings','team_id','football.teams','team_id',false),
-- squads, transfers, injuries
('football.team_squads','player_id','football.players','player_id',false),
('football.team_squads','season_id','football.seasons','season_id',false),
('football.team_squads','transfer_id','football.transfers','transfer_id',false),
('football.team_squads','position_id','core.types','type_id',false),
('football.team_squads','detailed_position_id','core.types','type_id',false),
('football.coach_teams','coach_id','football.coaches','coach_id',false),
('football.coach_teams','team_id','football.teams','team_id',false),
('football.coach_teams','position_id','core.types','type_id',false),
('football.transfers','player_id','football.players','player_id',false),
('football.transfers','type_id','core.types','type_id',false),
('football.transfers','from_team_id','football.teams','team_id',false),
('football.transfers','to_team_id','football.teams','team_id',false),
('football.transfers','position_id','core.types','type_id',false),
('football.transfers','detailed_position_id','core.types','type_id',false),
('football.transfer_rumours','player_id','football.players','player_id',false),
('football.transfer_rumours','position_id','core.types','type_id',false),
('football.transfer_rumours','from_team_id','football.teams','team_id',false),
('football.transfer_rumours','to_team_id','football.teams','team_id',false),
('football.transfer_rumours','type_id','core.types','type_id',false),
('football.sidelined','player_id','football.players','player_id',false),
('football.sidelined','team_id','football.teams','team_id',false),
('football.sidelined','season_id','football.seasons','season_id',false),
('football.sidelined','type_id','core.types','type_id',false),
-- fixtures
('football.fixtures','league_id','football.leagues','league_id',true),
('football.fixtures','season_id','football.seasons','season_id',true),
('football.fixtures','stage_id','football.stages','stage_id',false),
('football.fixtures','round_id','football.rounds','round_id',false),
('football.fixtures','group_id','football.groups','group_id',false),
('football.fixtures','aggregate_id','football.aggregates','aggregate_id',false),
('football.fixtures','state_id','core.states','state_id',false),
('football.fixtures','venue_id','football.venues','venue_id',false),
('football.fixtures','home_team_id','football.teams','team_id',false),
('football.fixtures','away_team_id','football.teams','team_id',false),
-- inside fixtures
('football.fixture_participants','team_id','football.teams','team_id',false),
('football.fixture_participants','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_scores','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_scores','team_id','football.teams','team_id',false),
('football.fixture_scores','type_id','core.types','type_id',false),
('football.fixture_periods','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_periods','type_id','core.types','type_id',false),
('football.fixture_events','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_events','period_id','football.fixture_periods','period_id',false),
('football.fixture_events','team_id','football.teams','team_id',false),
('football.fixture_events','type_id','core.types','type_id',false),
('football.fixture_events','sub_type_id','core.types','type_id',false),
('football.fixture_events','player_id','football.players','player_id',false),
('football.fixture_events','related_player_id','football.players','player_id',false),
('football.fixture_events','coach_id','football.coaches','coach_id',false),
('football.fixture_lineups','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_lineups','team_id','football.teams','team_id',false),
('football.fixture_lineups','player_id','football.players','player_id',false),
('football.fixture_lineups','type_id','core.types','type_id',false),
('football.fixture_lineups','position_id','core.types','type_id',false),
('football.fixture_lineups','detailed_position_id','core.types','type_id',false),
('football.fixture_lineup_details','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_lineup_details','lineup_id','football.fixture_lineups','lineup_id',false),
('football.fixture_lineup_details','team_id','football.teams','team_id',false),
('football.fixture_lineup_details','type_id','core.types','type_id',false),
('football.fixture_statistics','team_id','football.teams','team_id',false),
('football.fixture_statistics','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_statistics','type_id','core.types','type_id',false),
('football.fixture_formations','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_formations','team_id','football.teams','team_id',false),
('football.fixture_officials','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_officials','referee_id','football.referees','referee_id',false),
('football.fixture_officials','type_id','core.types','type_id',false),
('football.fixture_coaches','coach_id','football.coaches','coach_id',false),
('football.fixture_coaches','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_coaches','team_id','football.teams','team_id',false),
('football.fixture_tv_stations','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_tv_stations','tv_station_id','football.tv_stations','tv_station_id',false),
('football.fixture_tv_stations','country_id','core.countries','country_id',false),
('football.fixture_sidelined','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_sidelined','sidelined_id','football.sidelined','sidelined_id',false),
('football.fixture_sidelined','team_id','football.teams','team_id',false),
('football.fixture_weather','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_weather','venue_id','football.venues','venue_id',false),
('football.fixture_trends','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_trends','team_id','football.teams','team_id',false),
('football.fixture_trends','type_id','core.types','type_id',false),
('football.fixture_pressure','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_pressure','team_id','football.teams','team_id',false),
('football.fixture_ball_coordinates','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_xg','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_xg','team_id','football.teams','team_id',false),
('football.fixture_xg','type_id','core.types','type_id',false),
('football.fixture_xg_players','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_xg_players','player_id','football.players','player_id',false),
('football.fixture_xg_players','team_id','football.teams','team_id',false),
('football.fixture_xg_players','type_id','core.types','type_id',false),
('football.fixture_expected_lineups','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_expected_lineups','player_id','football.players','player_id',false),
('football.fixture_expected_lineups','team_id','football.teams','team_id',false),
('football.fixture_commentaries','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_commentaries','player_id','football.players','player_id',false),
('football.fixture_match_facts','fixture_id','football.fixtures','fixture_id',true),
('football.fixture_news','fixture_id','football.fixtures','fixture_id',false),
('football.fixture_news','league_id','football.leagues','league_id',false),
('football.fixture_news_lines','news_id','football.fixture_news','news_id',true),
('football.metadata','type_id','core.types','type_id',false),
-- standings / topscorers
('football.standings','season_id','football.seasons','season_id',true),
('football.standings','league_id','football.leagues','league_id',false),
('football.standings','team_id','football.teams','team_id',false),
('football.standings','stage_id','football.stages','stage_id',false),
('football.standings','round_id','football.rounds','round_id',false),
('football.standing_details','standing_id','football.standings','standing_id',true),
('football.standing_details','type_id','core.types','type_id',false),
('football.standing_form','standing_id','football.standings','standing_id',true),
('football.standing_form','fixture_id','football.fixtures','fixture_id',false),
('football.topscorers','season_id','football.seasons','season_id',true),
('football.topscorers','player_id','football.players','player_id',false),
('football.topscorers','team_id','football.teams','team_id',false),
('football.topscorers','type_id','core.types','type_id',false),
-- season statistics
('football.season_statistics','season_id','football.seasons','season_id',true),
('football.season_statistics','type_id','core.types','type_id',false),
('football.team_season_stats','season_id','football.seasons','season_id',false),
('football.team_season_stat_details','team_statistic_id','football.team_season_stats','team_statistic_id',true),
('football.team_season_stat_details','type_id','core.types','type_id',false),
('football.player_season_stats','player_id','football.players','player_id',false),
('football.player_season_stats','team_id','football.teams','team_id',false),
('football.player_season_stats','position_id','core.types','type_id',false),
('football.player_season_stat_details','player_statistic_id','football.player_season_stats','player_statistic_id',true),
('football.player_season_stat_details','type_id','core.types','type_id',false),
('football.coach_season_stats','coach_id','football.coaches','coach_id',false),
('football.coach_season_stats','team_id','football.teams','team_id',false),
('football.coach_season_stats','season_id','football.seasons','season_id',false),
('football.coach_season_stat_details','coach_statistic_id','football.coach_season_stats','coach_statistic_id',true),
('football.coach_season_stat_details','type_id','core.types','type_id',false),
('football.referee_season_stats','season_id','football.seasons','season_id',false),
('football.referee_season_stat_details','referee_statistic_id','football.referee_season_stats','referee_statistic_id',true),
('football.referee_season_stat_details','type_id','core.types','type_id',false),
-- predictions
('football.predictions','fixture_id','football.fixtures','fixture_id',true),
('football.predictions','type_id','core.types','type_id',false),
('football.live_predictions','fixture_id','football.fixtures','fixture_id',true),
('football.league_probabilities','league_id','football.leagues','league_id',false),
-- odds
('odds.prematch_odds','fixture_id','football.fixtures','fixture_id',true),
('odds.prematch_odds','market_id','odds.markets','market_id',false),
('odds.prematch_odds','bookmaker_id','odds.bookmakers','bookmaker_id',false),
('odds.inplay_odds','fixture_id','football.fixtures','fixture_id',true),
('odds.inplay_odds','market_id','odds.markets','market_id',false),
('odds.inplay_odds','bookmaker_id','odds.bookmakers','bookmaker_id',false),
('odds.premium_odds','fixture_id','football.fixtures','fixture_id',true),
('odds.premium_odds','market_id','odds.markets','market_id',false),
('odds.premium_odds','bookmaker_id','odds.bookmakers','bookmaker_id',false),
('odds.premium_odds_history','odd_id','odds.premium_odds','odd_id',true);

-- The fixture-context columns (league_id / season_id copied from the
-- fixture) are also referenced so they're indexed.
INSERT INTO raw.relationships (tbl, col, ref_tbl, ref_col, enforced)
SELECT c.table_schema || '.' || c.table_name, c.column_name,
       CASE c.column_name WHEN 'league_id' THEN 'football.leagues' ELSE 'football.seasons' END,
       c.column_name, false
FROM information_schema.columns c
WHERE c.table_schema IN ('football', 'odds')
  AND c.column_name IN ('league_id', 'season_id')
  AND EXISTS (SELECT 1 FROM information_schema.columns c2
              WHERE c2.table_schema = c.table_schema AND c2.table_name = c.table_name
                AND c2.column_name = 'fixture_id')
  AND c.table_name <> 'fixtures'
ON CONFLICT (tbl, col) DO NOTHING;


-- Build foreign keys and indexes from the map.
DO $$
DECLARE
    r        record;
    short    text;
    is_lead  boolean;
BEGIN
    FOR r IN SELECT * FROM raw.relationships ORDER BY tbl, col LOOP
        short := replace(r.tbl, '.', '_') || '_' || r.col;

        IF r.enforced THEN
            EXECUTE format(
                'ALTER TABLE %s ADD CONSTRAINT %I FOREIGN KEY (%I) REFERENCES %s (%I) '
                'ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED',
                r.tbl, left('fk_' || short, 63), r.col, r.ref_tbl, r.ref_col);
        END IF;

        -- Skip the index if an existing index already starts with this column.
        SELECT EXISTS (
            SELECT 1
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
            WHERE i.indrelid = r.tbl::regclass AND a.attname = r.col
        ) INTO is_lead;

        IF NOT is_lead THEN
            EXECUTE format('CREATE INDEX %I ON %s (%I)', left('ix_' || short, 63), r.tbl, r.col);
        END IF;
    END LOOP;
END $$;


/* ============================================================================
   TRIGGERS
   ========================================================================= */

-- Copy league_id / season_id from the fixture onto child rows.
-- Insert the fixture row before its children (same transaction is fine).
CREATE OR REPLACE FUNCTION football.fill_fixture_context() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.season_id IS NULL OR NEW.league_id IS NULL THEN
        SELECT f.league_id, f.season_id
          INTO NEW.league_id, NEW.season_id
          FROM football.fixtures f
         WHERE f.fixture_id = NEW.fixture_id;
    END IF;
    RETURN NEW;
END $$;

DO $$
DECLARE t record;
BEGIN
    FOR t IN
        SELECT c.table_schema, c.table_name
        FROM information_schema.columns c
        WHERE c.table_schema IN ('football', 'odds')
          AND c.table_name <> 'fixtures'
          AND c.column_name IN ('fixture_id', 'league_id', 'season_id')
        GROUP BY 1, 2
        HAVING count(*) = 3
    LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_fixture_context BEFORE INSERT OR UPDATE OF fixture_id ON %I.%I '
            'FOR EACH ROW EXECUTE FUNCTION football.fill_fixture_context()',
            t.table_schema, t.table_name);
    END LOOP;
END $$;

-- Keep fixtures.home_team_id / away_team_id in step with participants.
CREATE OR REPLACE FUNCTION football.sync_home_away() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.location = 'home' THEN
        UPDATE football.fixtures SET home_team_id = NEW.team_id
         WHERE fixture_id = NEW.fixture_id AND home_team_id IS DISTINCT FROM NEW.team_id;
    ELSIF NEW.location = 'away' THEN
        UPDATE football.fixtures SET away_team_id = NEW.team_id
         WHERE fixture_id = NEW.fixture_id AND away_team_id IS DISTINCT FROM NEW.team_id;
    END IF;
    RETURN NULL;
END $$;

CREATE TRIGGER trg_sync_home_away
AFTER INSERT OR UPDATE OF location, team_id ON football.fixture_participants
FOR EACH ROW EXECUTE FUNCTION football.sync_home_away();


/* ============================================================================
   DATA-QUALITY HELPER
   ========================================================================= */

-- IDs referenced somewhere but not yet in their own table.
-- Use it to decide what to fetch next:  SELECT * FROM raw.missing_references();
CREATE OR REPLACE FUNCTION raw.missing_references()
RETURNS TABLE (ref_tbl text, missing_id bigint, referenced_from text, times_referenced bigint)
LANGUAGE plpgsql AS $$
DECLARE r record;
BEGIN
    FOR r IN SELECT * FROM raw.relationships WHERE NOT enforced ORDER BY ref_tbl, tbl LOOP
        RETURN QUERY EXECUTE format(
            'SELECT %L::text, s.%I::bigint, %L::text, count(*)::bigint
               FROM %s s
              WHERE s.%I IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM %s t WHERE t.%I = s.%I)
              GROUP BY s.%I',
            r.ref_tbl, r.col, r.tbl || '.' || r.col, r.tbl, r.col,
            r.ref_tbl, r.ref_col, r.col, r.col);
    END LOOP;
END $$;


/* ============================================================================
   ANALYSIS LAYER
   Views read live data. Materialized views are pre-computed for speed:
   run  SELECT analysis.refresh_all();  after each load.

   The wide stat columns use Sportmonks developer_names. Every statistic is
   also kept in the all_stats JSON column, so nothing is lost if a name
   differs, e.g.  (all_stats->>'HIT_WOODWORK')::numeric
   ========================================================================= */

-- Finished-match states
CREATE OR REPLACE VIEW analysis.finished_states AS
SELECT state_id FROM core.states
WHERE developer_name IN ('FT', 'AET', 'FT_PEN', 'AWARDED');

-- The on-field referee for each match
CREATE OR REPLACE VIEW analysis.main_referee AS
SELECT DISTINCT ON (fo.fixture_id) fo.fixture_id, fo.referee_id
FROM football.fixture_officials fo
JOIN core.types t ON t.type_id = fo.type_id
WHERE t.developer_name = 'REFEREE' OR lower(t.name) = 'referee'
ORDER BY fo.fixture_id, fo.fixture_official_id;

-- One row per match with names, score and referee
CREATE OR REPLACE VIEW analysis.fixture_overview AS
SELECT
    f.fixture_id,
    f.league_id,   l.name  AS league,
    f.season_id,   s.name  AS season,
    f.stage_id,    stg.name AS stage,
    f.round_id,    r.name  AS round,
    f.starting_at,
    st.developer_name      AS state,
    (f.state_id IN (SELECT state_id FROM analysis.finished_states)) AS is_finished,
    f.home_team_id, ht.name AS home_team,
    f.away_team_id, at.name AS away_team,
    sc.home_goals,
    sc.away_goals,
    mr.referee_id,
    COALESCE(rf.display_name, rf.common_name, rf.name) AS referee,
    f.venue_id,    v.name  AS venue
FROM football.fixtures f
LEFT JOIN football.leagues  l   ON l.league_id  = f.league_id
LEFT JOIN football.seasons  s   ON s.season_id  = f.season_id
LEFT JOIN football.stages   stg ON stg.stage_id = f.stage_id
LEFT JOIN football.rounds   r   ON r.round_id   = f.round_id
LEFT JOIN core.states       st  ON st.state_id  = f.state_id
LEFT JOIN football.teams    ht  ON ht.team_id   = f.home_team_id
LEFT JOIN football.teams    at  ON at.team_id   = f.away_team_id
LEFT JOIN football.venues   v   ON v.venue_id   = f.venue_id
LEFT JOIN analysis.main_referee mr ON mr.fixture_id = f.fixture_id
LEFT JOIN football.referees rf  ON rf.referee_id = mr.referee_id
LEFT JOIN LATERAL (
    SELECT max(goals) FILTER (WHERE location = 'home') AS home_goals,
           max(goals) FILTER (WHERE location = 'away') AS away_goals
    FROM football.fixture_scores
    WHERE fixture_id = f.fixture_id AND description = 'CURRENT'
) sc ON true;


-- One row per team per match: result, key stats, cards, xG, all stats as JSON
CREATE MATERIALIZED VIEW analysis.team_match_stats AS
WITH base AS (
    SELECT f.fixture_id, f.league_id, f.season_id, f.starting_at, p.team_id,
           p.location,
           CASE WHEN p.team_id = f.home_team_id THEN f.away_team_id
                ELSE f.home_team_id END AS opponent_id
    FROM football.fixtures f
    JOIN football.fixture_participants p ON p.fixture_id = f.fixture_id
),
goals AS (
    SELECT fixture_id, location, max(goals) AS goals
    FROM football.fixture_scores
    WHERE description = 'CURRENT'
    GROUP BY 1, 2
),
stats AS (
    SELECT s.fixture_id, s.team_id,
        max(s.value) FILTER (WHERE t.developer_name = 'SHOTS_TOTAL')        AS shots,
        max(s.value) FILTER (WHERE t.developer_name = 'SHOTS_ON_TARGET')    AS shots_on_target,
        max(s.value) FILTER (WHERE t.developer_name = 'CORNERS')            AS corners,
        max(s.value) FILTER (WHERE t.developer_name = 'FOULS')              AS fouls,
        max(s.value) FILTER (WHERE t.developer_name = 'OFFSIDES')           AS offsides,
        max(s.value) FILTER (WHERE t.developer_name = 'BALL_POSSESSION')    AS possession_pct,
        max(s.value) FILTER (WHERE t.developer_name = 'PASSES')             AS passes,
        max(s.value) FILTER (WHERE t.developer_name = 'SUCCESSFUL_PASSES')  AS successful_passes,
        max(s.value) FILTER (WHERE t.developer_name = 'TACKLES')            AS tackles,
        max(s.value) FILTER (WHERE t.developer_name = 'INTERCEPTIONS')      AS interceptions,
        max(s.value) FILTER (WHERE t.developer_name = 'SAVES')              AS saves,
        max(s.value) FILTER (WHERE t.developer_name = 'ATTACKS')            AS attacks,
        max(s.value) FILTER (WHERE t.developer_name = 'DANGEROUS_ATTACKS')  AS dangerous_attacks,
        max(s.value) FILTER (WHERE t.developer_name = 'FREE_KICKS')         AS free_kicks,
        jsonb_object_agg(t.developer_name, s.value)
            FILTER (WHERE t.developer_name IS NOT NULL)                     AS all_stats
    FROM football.fixture_statistics s
    JOIN core.types t ON t.type_id = s.type_id
    GROUP BY 1, 2
),
cards AS (
    SELECT e.fixture_id, e.team_id,
        count(*) FILTER (WHERE t.developer_name = 'YELLOWCARD')                         AS yellow_cards,
        count(*) FILTER (WHERE t.developer_name = 'YELLOWREDCARD')                      AS second_yellows,
        count(*) FILTER (WHERE t.developer_name IN ('REDCARD', 'YELLOWREDCARD'))         AS red_cards,
        count(*) FILTER (WHERE t.developer_name IN ('PENALTY', 'MISSED_PENALTY'))        AS penalties_awarded,
        count(*) FILTER (WHERE t.developer_name LIKE 'VAR%')                             AS var_events,
        count(*) FILTER (WHERE t.developer_name IN ('YELLOWCARD','REDCARD','YELLOWREDCARD')
                           AND e.coach_id IS NOT NULL)                                   AS coach_cards
    FROM football.fixture_events e
    JOIN core.types t ON t.type_id = e.type_id
    WHERE e.rescinded IS NOT TRUE
    GROUP BY 1, 2
),
xg AS (
    SELECT x.fixture_id, x.team_id,
        max(x.value) FILTER (WHERE t.developer_name = 'EXPECTED_GOALS') AS xg
    FROM football.fixture_xg x
    JOIN core.types t ON t.type_id = x.type_id
    GROUP BY 1, 2
)
SELECT
    b.fixture_id, b.league_id, b.season_id, b.starting_at,
    b.team_id, b.opponent_id, b.location,
    (b.location = 'home') AS is_home,
    g.goals  AS goals_for,
    og.goals AS goals_against,
    CASE WHEN g.goals > og.goals THEN 'W' WHEN g.goals < og.goals THEN 'L'
         WHEN g.goals = og.goals THEN 'D' END AS result,
    xg.xg, oxg.xg AS xg_against,
    st.shots, st.shots_on_target, st.corners, st.fouls, ost.fouls AS fouls_against,
    st.offsides, st.possession_pct, st.passes, st.successful_passes,
    st.tackles, st.interceptions, st.saves, st.attacks, st.dangerous_attacks, st.free_kicks,
    COALESCE(c.yellow_cards, 0)      AS yellow_cards,
    COALESCE(c.second_yellows, 0)    AS second_yellows,
    COALESCE(c.red_cards, 0)         AS red_cards,
    COALESCE(c.penalties_awarded, 0) AS penalties_awarded,
    COALESCE(c.var_events, 0)        AS var_events,
    COALESCE(c.coach_cards, 0)       AS coach_cards,
    st.all_stats
FROM base b
LEFT JOIN goals g   ON g.fixture_id = b.fixture_id  AND g.location = b.location
LEFT JOIN goals og  ON og.fixture_id = b.fixture_id AND og.location <> b.location
LEFT JOIN stats st  ON st.fixture_id = b.fixture_id  AND st.team_id = b.team_id
LEFT JOIN stats ost ON ost.fixture_id = b.fixture_id AND ost.team_id = b.opponent_id
LEFT JOIN cards c   ON c.fixture_id = b.fixture_id   AND c.team_id = b.team_id
LEFT JOIN xg        ON xg.fixture_id = b.fixture_id  AND xg.team_id = b.team_id
LEFT JOIN xg oxg    ON oxg.fixture_id = b.fixture_id AND oxg.team_id = b.opponent_id;

CREATE UNIQUE INDEX ux_team_match_stats ON analysis.team_match_stats (fixture_id, team_id);
CREATE INDEX ix_team_match_stats_team_season ON analysis.team_match_stats (team_id, season_id);
CREATE INDEX ix_team_match_stats_league_season ON analysis.team_match_stats (league_id, season_id);


-- One row per player per match: minutes, rating, key stats, all stats as JSON
CREATE MATERIALIZED VIEW analysis.player_match_stats AS
WITH d AS (
    SELECT ld.lineup_id,
        max(ld.value) FILTER (WHERE t.developer_name = 'MINUTES_PLAYED')   AS minutes,
        max(ld.value) FILTER (WHERE t.developer_name = 'RATING')           AS rating,
        max(ld.value) FILTER (WHERE t.developer_name = 'GOALS')            AS goals,
        max(ld.value) FILTER (WHERE t.developer_name = 'ASSISTS')          AS assists,
        max(ld.value) FILTER (WHERE t.developer_name = 'SHOTS_TOTAL')      AS shots,
        max(ld.value) FILTER (WHERE t.developer_name = 'SHOTS_ON_TARGET')  AS shots_on_target,
        max(ld.value) FILTER (WHERE t.developer_name = 'PASSES')           AS passes,
        max(ld.value) FILTER (WHERE t.developer_name = 'ACCURATE_PASSES')  AS accurate_passes,
        max(ld.value) FILTER (WHERE t.developer_name = 'KEY_PASSES')       AS key_passes,
        max(ld.value) FILTER (WHERE t.developer_name = 'TACKLES')          AS tackles,
        max(ld.value) FILTER (WHERE t.developer_name = 'INTERCEPTIONS')    AS interceptions,
        max(ld.value) FILTER (WHERE t.developer_name = 'DUELS_WON')        AS duels_won,
        max(ld.value) FILTER (WHERE t.developer_name = 'FOULS')            AS fouls_committed,
        max(ld.value) FILTER (WHERE t.developer_name = 'FOULS_DRAWN')      AS fouls_drawn,
        max(ld.value) FILTER (WHERE t.developer_name = 'SAVES')            AS saves,
        jsonb_object_agg(t.developer_name, ld.value)
            FILTER (WHERE t.developer_name IS NOT NULL)                    AS all_stats
    FROM football.fixture_lineup_details ld
    JOIN core.types t ON t.type_id = ld.type_id
    GROUP BY ld.lineup_id
),
cards AS (
    SELECT e.fixture_id, e.player_id,
        count(*) FILTER (WHERE t.developer_name = 'YELLOWCARD')                 AS yellow_cards,
        count(*) FILTER (WHERE t.developer_name IN ('REDCARD','YELLOWREDCARD')) AS red_cards
    FROM football.fixture_events e
    JOIN core.types t ON t.type_id = e.type_id
    WHERE e.rescinded IS NOT TRUE AND e.player_id IS NOT NULL
    GROUP BY 1, 2
),
xg AS (
    SELECT x.fixture_id, x.player_id,
        max(x.value) FILTER (WHERE t.developer_name = 'EXPECTED_GOALS') AS xg
    FROM football.fixture_xg_players x
    JOIN core.types t ON t.type_id = x.type_id
    GROUP BY 1, 2
)
SELECT
    l.lineup_id, l.fixture_id, l.league_id, l.season_id, f.starting_at,
    l.team_id, l.player_id, l.player_name, l.position_id, l.jersey_number,
    (lt.developer_name = 'LINEUP') AS started,
    d.minutes, d.rating, d.goals, d.assists, xg.xg,
    d.shots, d.shots_on_target, d.passes, d.accurate_passes, d.key_passes,
    d.tackles, d.interceptions, d.duels_won, d.fouls_committed, d.fouls_drawn, d.saves,
    COALESCE(c.yellow_cards, 0) AS yellow_cards,
    COALESCE(c.red_cards, 0)    AS red_cards,
    d.all_stats
FROM football.fixture_lineups l
JOIN football.fixtures f ON f.fixture_id = l.fixture_id
LEFT JOIN core.types lt  ON lt.type_id = l.type_id
LEFT JOIN d              ON d.lineup_id = l.lineup_id
LEFT JOIN cards c        ON c.fixture_id = l.fixture_id AND c.player_id = l.player_id
LEFT JOIN xg             ON xg.fixture_id = l.fixture_id AND xg.player_id = l.player_id;

CREATE UNIQUE INDEX ux_player_match_stats ON analysis.player_match_stats (lineup_id);
CREATE INDEX ix_player_match_stats_player_season ON analysis.player_match_stats (player_id, season_id);
CREATE INDEX ix_player_match_stats_fixture ON analysis.player_match_stats (fixture_id);


-- One row per finished match with its referee: discipline split home/away
CREATE MATERIALIZED VIEW analysis.referee_match_discipline AS
SELECT
    f.fixture_id, f.league_id, f.season_id, f.starting_at,
    mr.referee_id,
    f.home_team_id, f.away_team_id,
    h.goals_for  AS home_goals,       a.goals_for  AS away_goals,
    h.yellow_cards AS home_yellows,   a.yellow_cards AS away_yellows,
    h.second_yellows AS home_second_yellows, a.second_yellows AS away_second_yellows,
    h.red_cards  AS home_reds,        a.red_cards  AS away_reds,
    h.fouls      AS home_fouls,       a.fouls      AS away_fouls,
    h.penalties_awarded AS home_penalties, a.penalties_awarded AS away_penalties,
    h.coach_cards + a.coach_cards                AS coach_cards,
    h.var_events + a.var_events                  AS var_events,
    h.yellow_cards + a.yellow_cards              AS total_yellows,
    h.red_cards + a.red_cards                    AS total_reds,
    h.fouls + a.fouls                            AS total_fouls,
    h.penalties_awarded + a.penalties_awarded    AS total_penalties
FROM football.fixtures f
JOIN analysis.main_referee mr      ON mr.fixture_id = f.fixture_id
LEFT JOIN analysis.team_match_stats h ON h.fixture_id = f.fixture_id AND h.team_id = f.home_team_id
LEFT JOIN analysis.team_match_stats a ON a.fixture_id = f.fixture_id AND a.team_id = f.away_team_id
WHERE f.state_id IN (SELECT state_id FROM analysis.finished_states);

CREATE UNIQUE INDEX ux_referee_match_discipline ON analysis.referee_match_discipline (fixture_id);
CREATE INDEX ix_referee_match_discipline_ref ON analysis.referee_match_discipline (referee_id, season_id);
CREATE INDEX ix_referee_match_discipline_league ON analysis.referee_match_discipline (league_id, season_id);


-- Referee per league-season, compared with that league-season's average.
-- *_index = referee average / league average (1.20 = 20% above league).
CREATE OR REPLACE VIEW analysis.referee_season_summary AS
WITH lg AS (
    SELECT league_id, season_id,
           avg(total_yellows)   AS lg_yellows,
           avg(total_reds)      AS lg_reds,
           avg(total_fouls)     AS lg_fouls,
           avg(total_penalties) AS lg_penalties
    FROM analysis.referee_match_discipline
    GROUP BY 1, 2
)
SELECT
    m.referee_id,
    COALESCE(r.display_name, r.common_name, r.name) AS referee,
    m.league_id, l.name AS league,
    m.season_id, s.name AS season,
    count(*)                                            AS matches,
    round(avg(m.total_yellows), 2)                      AS yellows_pm,
    round(avg(m.total_reds), 3)                         AS reds_pm,
    round(avg(m.total_fouls), 1)                        AS fouls_pm,
    round(avg(m.total_penalties), 3)                    AS penalties_pm,
    round(avg(m.home_yellows), 2)                       AS home_yellows_pm,
    round(avg(m.away_yellows), 2)                       AS away_yellows_pm,
    round(avg(m.home_yellows - m.away_yellows), 2)      AS home_minus_away_yellows,
    round(avg(m.home_fouls - m.away_fouls), 2)          AS home_minus_away_fouls,
    round(sum(m.total_yellows)::numeric / NULLIF(sum(m.total_fouls), 0), 3) AS yellows_per_foul,
    round(avg((m.home_goals > m.away_goals)::int), 3)   AS home_win_rate,
    round(avg(m.total_yellows) / NULLIF(max(lg.lg_yellows), 0), 2)     AS yellows_index,
    round(avg(m.total_fouls)   / NULLIF(max(lg.lg_fouls), 0), 2)       AS fouls_index,
    round(avg(m.total_penalties) / NULLIF(max(lg.lg_penalties), 0), 2) AS penalties_index
FROM analysis.referee_match_discipline m
JOIN lg ON lg.league_id = m.league_id AND lg.season_id = m.season_id
LEFT JOIN football.referees r ON r.referee_id = m.referee_id
LEFT JOIN football.leagues  l ON l.league_id  = m.league_id
LEFT JOIN football.seasons  s ON s.season_id  = m.season_id
GROUP BY m.referee_id, r.display_name, r.common_name, r.name,
         m.league_id, l.name, m.season_id, s.name;


-- Refresh all materialized views in dependency order.
CREATE OR REPLACE FUNCTION analysis.refresh_all() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW analysis.team_match_stats;
    REFRESH MATERIALIZED VIEW analysis.player_match_stats;
    REFRESH MATERIALIZED VIEW analysis.referee_match_discipline;
    ANALYZE analysis.team_match_stats;
    ANALYZE analysis.player_match_stats;
    ANALYZE analysis.referee_match_discipline;
END $$;


/* ============================================================================
   TABLE DESCRIPTIONS (visible in DBeaver / pgAdmin / \d+)
   ========================================================================= */

COMMENT ON TABLE core.types IS 'Every coded value: event, statistic, position, official role, lineup type. Filter by developer_name.';
COMMENT ON TABLE football.fixtures IS 'One row per match. Source: /football/fixtures.';
COMMENT ON TABLE football.fixture_events IS 'Goals, cards, subs, VAR, penalties. Include: events / timeline. rescinded = later cancelled.';
COMMENT ON TABLE football.fixture_lineup_details IS 'Player match stats. Include: lineups.details. value = numeric headline of data.';
COMMENT ON TABLE football.fixture_statistics IS 'Team match stats. Include: statistics.';
COMMENT ON TABLE football.fixture_officials IS 'Match officials and their role (type_id). Include: referees.';
COMMENT ON TABLE football.fixture_xg IS 'Team expected metrics. Include: xGFixture.';
COMMENT ON TABLE football.fixture_xg_players IS 'Player expected metrics. Include: lineups.xGLineup.';
COMMENT ON TABLE football.fixture_pressure IS 'Pressure Index per minute. Include: pressure.';
COMMENT ON TABLE football.fixture_ball_coordinates IS 'Ball position samples. Include: ballCoordinates. Very large.';
COMMENT ON TABLE odds.premium_odds_history IS 'Price movement for premium odds. Include: history on premium odds.';
COMMENT ON TABLE raw.relationships IS 'Map of every reference between tables. Drives FKs, indexes and raw.missing_references().';
COMMENT ON MATERIALIZED VIEW analysis.team_match_stats IS 'One row per team per match. Refresh with analysis.refresh_all().';
COMMENT ON MATERIALIZED VIEW analysis.player_match_stats IS 'One row per player per match. Refresh with analysis.refresh_all().';
COMMENT ON MATERIALIZED VIEW analysis.referee_match_discipline IS 'One row per finished match with its referee. Refresh with analysis.refresh_all().';
COMMENT ON VIEW analysis.referee_season_summary IS 'Referee averages per league-season with league-relative indexes.';

COMMIT;
