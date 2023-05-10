drop schema if exists public cascade;
create schema if not exists public;

create table users
(
    id           serial primary key,
    login        text not null,
    password     text not null,
    access_level int  not null default 0
);

insert into users (login, password, access_level)
values ('admin', 'admin', 2),
       ('user', 'user', 1);

create table workers
(
    id   serial primary key,
    name text not null
);

create table documents
(
    id             serial primary key,
    worker         integer references workers (id),
    name           text not null,
    date_of_apply  date not null,
    date_of_return date not null
);

insert into workers (name)
values ('Иванов Иван Иванович'),
       ('Петров Петр Петрович'),
       ('Гамалеева Софья Николаевна'),
       ('Сидоров Сидор Сидорович'),
       ('Гунько Андрей Васильевич'),
       ('Наталья Морская Пехота');

insert into documents (worker, name, date_of_apply, date_of_return)
values (1, 'Роспись о доставке', '2019-01-13', '2019-02-01'),
       (2, 'Договор подряда', '2022-01-13', '2019-02-01'),
       (5, 'Договор оказания услуг', '2019-01-13', '2019-02-01'),
       (5, 'Договор комиссии', '2019-01-13', '2019-02-01'),
       (3, 'Договор о найме', '2023-04-21', '2019-05-01'),
       (2, 'Договор страхования', '2019-01-13', '2019-02-01'),
       (6, 'Договор субаренды', '2019-01-13', '2019-02-01'),
       (4, 'Роспись о доставке', '2019-01-13', '2019-02-01');
