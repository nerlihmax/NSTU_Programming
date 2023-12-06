drop schema if exists public cascade;
create schema if not exists public;

create table users
(
    user_id       text primary key,
    name          text,
    email         text not null,
    password_hash text,
    auth_provider text not null default 'local'
);

create table refresh_tokens
(
    id         serial primary key,
    user_id    text   not null references users (user_id) on delete cascade,
    client_id  text   not null,
    token      text   not null unique,
    expires_at bigint not null
);

create table hotels
(
    id      serial primary key,
    name    text not null,
    city    text not null,
    address text not null,
    stars   int  not null
);

create table employees
(
    id       serial primary key,
    user_id  text not null references users (user_id) on delete cascade,
    hotel_id int  not null references hotels (id) on delete cascade
);

create table rooms
(
    room_id   serial primary key,
    room_type text not null,
    price     int  not null,
    hotel_id  int references hotels (id)
);

create table reservations
(
    reservation_id serial primary key,
    guest          text not null references users (user_id) on delete cascade,
    arrival_date   datetime not null,
    departure_date datetime not null,
    room_id        int references rooms (room_id)
);

insert into users (user_id, name, email, password_hash, auth_provider)
values ('ebd376eb', 'Иванов Иван Иванович', 'ivanov@hotel.kheynov.ru',
        '$2a$12$Zxs4d/nFyQdC1P/Dt53qze9vOkDxcZ/9OnRqnUFmwiZlUpF2DEQ6a', 'local'),
       ('a1f8f991', 'Петров Петр Петрович', 'petrov@hotel.kheynov.ru',
        '$2a$12$PWP6J0N/nDoVv3Ik3/lLd.kkg.e4KscLsNcD7cDR8W0JFtWwKY.u6', 'local'),
       ('21dde366', 'Сидоров Сидор Сидорович', 'sidorov@hotel.kheynov.ru',
        '$2a$12$VK0uWqfG.J47KxrbRQhyyelDh4SbupY1ErtEJxLBuo26FmlO5D/sW', 'local'),
       ('32adf77a', 'Александров Александр Александрович', 'alexandrov@hotel.kheynov.ru',
        '$2a$12$BP6o14ecItdn0r0bcBohOuE6E0mb08ar7ZLMuubyx47HDhQW49Y.e', 'local'),
       ('4e0dc99d', 'Андреев Андрей Андреевич', 'andreev@hotel.kheynov.ru',
        '$2a$12$3.DqrMqgsJZACI1yBhN0o.i6J/S16z6.b2QebcCEwCYJrSUpjJ2WW', 'local'),
       ('f8d5d633', 'Алексеев Алексей Алексеевич', 'alexeev@hotel.kheynov.ru',
        '$2a$12$snyaFn.7Or93Fv6a4KjlHuscUv6kjwVTmvJIj8mKx68IYDe7NG5I6', 'local'),

       ('303bd871', 'Антонов Антон Антонович', 'antonov@hotel.kheynov.ru',
        '$2a$12$qyiduzaStn.wsWVTjMs3qe1fIBAe1rFMnEaaATv9vBjIMzK4QSkb2', 'local');


-- insert into employees (user_id, hotel_id)
-- values ('ebd376eb', 1),
--        ('a1f8f991', 2),
--        ('21dde366', 3),
--        ('32adf77a', 4),
--        ('4e0dc99d', 5),
--        ('f8d5d633', 6),
--        ('303bd871', 7);

