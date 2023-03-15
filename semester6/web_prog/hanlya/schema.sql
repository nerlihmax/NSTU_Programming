drop schema if exists public cascade;
create schema if not exists public;

create table reader
(
    id   serial primary key,
    name text not null
);

create table issued_books
(
    id             serial primary key,
    name           text not null,
    reader         int references reader (id),
    date_of_issue  date not null,
    date_of_return date not null
);

insert into reader (name)
values ('Дмитрий Иванов'),
       ('Иван Петров'),
       ('Елена Малышева'),
       ('Алексей Сергеевич');

insert into issued_books
    (name, reader, date_of_issue, date_of_return)
values ('Гарри Поттер и кубок огня', 1, '2021-05-18', '2022-01-01'),
       ('Марсианин', 2, '2022-01-01', '2023-05-11'),
       ('Сумерки', 3, '2021-05-18', '2022-12-30'),
       ('Война и мир', 4, '2021-05-18', '2022-07-01'),
       ('Гарри Поттер и узник Азкабана', 1, '2021-01-09', '2022-01-01');

create table users
(
    id           serial primary key,
    login        text not null,
    password     text not null,
    access_level int  not null default 0
);

insert into users (login, password, access_level)
values ('admin', '12345678', 2),
       ('user', '12345678', 1);
