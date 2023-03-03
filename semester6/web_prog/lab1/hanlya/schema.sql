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
    date_of_issue  date check ( date_of_issue < current_date ),
    date_of_return timestamp
);

insert into reader (name)
values ('Дмитрий Иванов'),
       ('Иван Петров'),
       ('Елена Малышева'),
       ('Алексей Сергеевич');

insert into issued_books
    (name, reader, date_of_issue, date_of_return)
values ('Гарри Поттер и кубок огня', 1, '2021-05-18', '2022-01-01 12:00:00'),
       ('Марсианин', 2, '2022-01-01', null),
       ('Сумерки', 3, '2021-05-18', null),
       ('Война и мир', 4, '2021-05-18', '2019-01-01 12:00:00'),
       ('Гарри Поттер и узник Азкабана', 1, '2021-01-09', '2022-01-01 12:00:00');
