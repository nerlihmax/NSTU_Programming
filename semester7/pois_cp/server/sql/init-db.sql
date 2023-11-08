drop schema if exists public cascade;
create schema if not exists public;

create table users
(
    user_id       text primary key,
    name          text,
    email         text not null,
    password_hash text,
    auth_provider text not null default 'local',
    address       text
);

create table refresh_tokens
(
    id         serial primary key,
    user_id    text   not null references users (user_id) on delete cascade,
    client_id  text   not null,
    token      text   not null unique,
    expires_at bigint not null
);

create table cinemas
(
    id      text primary key not null default gen_random_uuid(),
    name    text             not null,
    city    text             not null,
    address text
);

create table employees
(
    id        text primary key default gen_random_uuid(),
    full_name text not null,
    cinema_id text not null references cinemas (id) on delete cascade
);

create table films
(
    id       text primary key default gen_random_uuid(),
    name     text not null,
    duration int  not null,
    image    text
);

create table timetables
(
    id        text primary key default gen_random_uuid(),
    film_id   text      not null references films (id) on DELETE cascade,
    cinema_id text      not null references cinemas (id) on delete cascade,
    price     int       not null,
    time      timestamp not null
);

create table bookings
(
    id          text primary key default gen_random_uuid(),
    employee_id text not null references employees (id) on delete cascade,
    booked_film text not null references timetables (id)
);
