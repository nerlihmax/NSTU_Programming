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
values ('admin', '12345678', 2),
       ('operator', '12345678', 1);

create table operation
(
    id   serial primary key,
    name text not null
);

create table technological_map
(
    id        serial primary key,
    name      text not null,
    operation int references operation (id),
    duration  int
);

insert into operation (name)
values ('Фрезеровка'),
       ('Токарная обработка'),
       ('Полировка'),
       ('Шлифовка'),
       ('Снятие фаски'),
       ('Дефектовка');

insert into technological_map (name, operation, duration)
values ('Шпинель', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Ручка', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Гайка', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Шайба', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Болт', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Шуруп', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Винт', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Шестерёнка', (select id from operation order by random() limit 1), (random() * 200 + 15)),
       ('Крепеж', (select id from operation order by random() limit 1), (random() * 200 + 15));

SELECT
            operation.name,
            count(*)
        from technological_map
        inner join operation on technological_map.operation = operation.id
        group by operation.name;