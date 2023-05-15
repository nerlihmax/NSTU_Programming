drop schema if exists public cascade;
create schema if not exists public;

create table country
(
    id   serial primary key,
    name text not null
);

insert into country
values (1, 'Аргентина'),
       (2, 'Китай'),
       (3, 'Россия'),
       (4, 'Нигерия'),
       (5, 'Португалия'),
       (6, 'Беларусь'),
       (7, 'Ирландия'),
       (8, 'Япония'),
       (9, 'Ангола'),
       (10, 'Казахстан');

create table provider
(
    id         serial primary key,
    name       text not null,
    country_id int references country (id)
);

insert into provider (name, country_id)
values ('Andalax', 9),
       ('Cardguard', 4),
       ('Holdlamis', 8),
       ('Bitchip', 2),
       ('Tin', 9),
       ('Holdlamis', 10),
       ('Home Ing', 4),
       ('Y-find', 7),
       ('Zathin', 1),
       ('Alphazap', 10);

create table furniture
(
    id          serial primary key,
    name        text not null,
    provider_id int references provider (id),
    price       int  not null check ( price > 0 )
);

insert into furniture (name, provider_id, price)
values ('Sonair', 1, 100),
       ('Subin', 2, 12),
       ('Viva', 3, 450),
       ('Wrapsafe', 4, 235),
       ('Voltsillam', 5, 1000),
       ('Solarbreeze', 6, 611),
       ('Opela', 7, 284),
       ('Alphazap', 8, 638),
       ('Stronghold', 9, 1695),
       ('Hatity', 10, 646);

create table users
(
    id    serial primary key,
    login text not null,
    pass  text not null,
    level int  not null check ( level = 1 or level = 2 )
);

insert into users (login, pass, level)
values ('admin', '12345678', 1),
       ('moderator', '12345678', 2);

create table city
(
    id         serial primary key,
    name       text not null,
    country_id int references country (id)
);

insert into city (country_id, name)
values (1, 'Буэнос-Айрес'),
       (1, 'Кордова'),
       (1, 'Росарио'),
       (2, 'Пекин'),
       (2, 'Шанхай'),
       (2, 'Чунцин'),
       (3, 'Москва'),
       (3, 'Санкт-Петербург'),
       (3, 'Новосибирск'),
       (4, 'Лагос'),
       (4, 'Кано'),
       (4, 'Ибадан'),
       (5, 'Лиссабон'),
       (5, 'Порту'),
       (5, 'Брага'),
       (6, 'Минск'),
       (6, 'Гомель'),
       (6, 'Могилев'),
       (7, 'Дублин'),
       (7, 'Корк'),
       (7, 'Лимерик'),
       (8, 'Токио'),
       (8, 'Киото'),
       (8, 'Осака'),
       (9, 'Луанда'),
       (9, 'Уила'),
       (9, 'Намья'),
       (10, 'Алматы'),
       (10, 'Нур-Султан'),
       (10, 'Шымкент');