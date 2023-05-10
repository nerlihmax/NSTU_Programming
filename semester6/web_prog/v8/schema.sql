drop schema if exists public cascade;
create schema public

    create table position
    (
        id   serial primary key,
        name text unique
    )

    create table degree
    (
        id   serial primary key,
        name text unique
    )

    create table courses
    (
        id   serial primary key,
        name text unique
    )

    create table teachers
    (
        id          serial primary key,
        position    int references position (id),
        degree      int references degree (id),
        courses     int references courses (id),
        surname     text,
        room_number int
    );

insert into position(name)
values ('Аспирант'),
       ('Доцент'),
       ('Профессор'),
       ('Ректор');

insert into degree(name)
values ('ктн. '),
       ('Старший преподаватель'),
       ('Доктор наук');

insert into courses(name)
values ('Web-программирование'),
       ('Клиент-серверный приложения'),
       ('Компиляторы'),
       ('Экономика'),
       ('БЖД');

insert into teachers
    (position, degree, courses, surname, room_number)
values (1, 1, 1, 'Иванов', 412),
       (2, 2, 2, 'Викторов', 138),
       (3, 1, 3, 'Попов', 213),
       (4, 2, 4, 'Сидоров', 321),
       (1, 1, 5, 'Пистолетов', 128);

select *
from teachers;

create table users
(
    id           serial primary key,
    login        text not null,
    password     text not null,
    access_level int  not null default 0
);

insert into users (login, password, access_level)
values ('admin', 'qwertyqwerty', 2),
       ('user', '123123', 1);