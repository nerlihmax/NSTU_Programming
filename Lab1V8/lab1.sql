drop schema if exists v8 cascade;
create schema v8;

create type v8.payment_state as ENUM ('PAID', 'PARTIALLY PAID', 'UNPAID');

create table if not exists v8.cities
(
    id   serial primary key,
    name text unique not null
);

create table if not exists v8.lodgers
(
    id          serial primary key,
    first_name  varchar(30) not null,
    second_name varchar(30) not null
);

create table if not exists v8.repairers
(
    id            serial primary key,
    first_name    varchar(30)                       not null,
    second_name   varchar(30)                       not null,
    date_of_birth date                              not null,
    city_id       integer references v8.cities (id) not null
);

create table if not exists v8.repairments
(
    id                  serial primary key,
    type                text                                 not null,
    repairer_id         integer references v8.repairers (id) not null,
    repairment_duration interval
);

create table if not exists v8.repair_facilities
(
    id              serial primary key,
    lodger_id       integer references v8.lodgers (id),
    repairment_date date,
    price           integer not null,
    payment_state   v8.payment_state,
    repairment_id   integer references v8.repairments (id)
);


insert into v8.lodgers (first_name, second_name)
values ('Василий', 'Абобович'),
       ('Ильдар', 'Виноградов'),
       ('Лев', 'Калинин'),
       ('Белла', 'Капустина'),
       ('Дана', 'Соломина'),
       ('Стелла', 'Бородина'),
       ('Адам', 'Александров'),
       ('Ника', 'Воронина'),
       ('Карина', 'Ибрагимова'),
       ('Ахмет', 'Логинов')
on conflict do nothing;

insert into v8.cities(name)
values ('Нижний Новгород'),
       ('Москва'),
       ('Екатиринбург'),
       ('Санкт-Петербург'),
       ('Новосибирск'),
       ('Бердск'),
       ('Ульяновск'),
       ('Барнаул'),
       ('Владивосток'),
       ('Сочи'),
       ('Искитим')
on conflict do nothing;

insert into v8.repairers (first_name, second_name, date_of_birth, city_id)
values ('Виктор', 'Ватов', '1955-01-15', 1),
       ('Любовь', 'Муравьёва', '1981-07-28', 2),
       ('Наталья', 'Ларионова', '1962-12-15', 3),
       ('Зоя', 'Измайлова', '1973-02-24', 1),
       ('Карп', 'Воронин', '1993-12-04', 5),
       ('Пётр', 'Боголюбов', '1997-10-11', 6),
       ('Иван', 'Петров', '2000-06-05', 9),
       ('Сергей', 'Беляков', '1979-05-18', 5),
       ('Евгений', 'Крюков', '1977-09-11', 8),
       ('Дмитрий', 'Долгоруков', '1999-01-25', 4)
on conflict do nothing;

insert into v8.repairments (type, repairer_id, repairment_duration)
values ('Замена окон', 2, '1 week'),
       ('Окрашивание дверей', 1, '1 week'),
       ('Поклеивание обоев', 4, '3 hours'),
       ('Замена ламината', 6, '2 weeks'),
       ('Установка телевизора', 5, '1 hour'),
       ('Удаление осиного гнезда', 7, '2 hours'),
       ('Изоляция дверных проёмов', 8, '2 days'),
       ('Сборка шкафа-купе', 5, '5 hours'),
       ('Ремонт сантехники', 9, '1 hour'),
       ('Замена замка на входной двери', 1, '3 hours')
on conflict do nothing;

insert into v8.repair_facilities (lodger_id, repairment_date, price, payment_state, repairment_id)
values (1, '2020-01-22', 5000, 'PAID', 1),
       (2, '2021-02-23', 6000, 'PAID', 2),
       (4, '2017-03-12', 10000, 'PAID', 3),
       (5, '2018-04-01', 2000, 'PAID', 4),
       (6, '2022-05-12', 2000, 'PAID', 5),
       (9, '2016-06-28', 3000, 'PAID', 6),
       (8, '2014-07-04', 5000, 'PAID', 7),
       (7, '2022-03-06', 15000, 'PAID', 8),
       (10, '2020-09-08', 1500, 'PAID', 9),
       (2, '2012-11-22', 5000, 'PAID', 10)
on conflict do nothing;


create or replace view v8.repairment_info as
select l.second_name         as lodger_surname,
       r.type                as repairment_type,
       rr.second_name        as repairer_surname,
       city.name             as city,
       r.repairment_duration as repairment_duration,
       rf.repairment_date    as repairment_date,
       rf.price              as price
from v8.repair_facilities as rf
         inner join v8.lodgers l on l.id = rf.lodger_id
         inner join v8.repairments r on r.id = rf.repairment_id
         inner join v8.repairers rr on rr.id = r.repairer_id
         inner join v8.cities city on city.id = rr.city_id;