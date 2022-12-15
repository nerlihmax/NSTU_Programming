drop schema if exists public cascade;
create schema public;

create type type_method as enum ('cash','cashless');
create type type_item as enum ('fridge', 'vacuum_cleaner', 'coffee_maker', 'hairdryer', 'multicooker', 'microwave_oven');

create table city
(
    id   serial primary key,
    name varchar not null
);

create table item
(
    id   serial primary key,
    type type_item                not null,
    cost integer check (cost > 0) not null
);

create table service
(
    id      serial not null primary key,
    name    text   not null,
    city_id integer references city (id),
    items   type_item[]
);

create table master
(
    id         serial  not null primary key,
    name       varchar not null,
    surname    varchar not null,
    city_id    integer references city (id),
    start_work date,
    birthday   date check (age(birthday) > interval '18 years'),
    service_id integer references service (id)
);

create table repair_order
(
    id                  serial                          not null primary key,
    date_of_application date                            not null,
    date_of_completion  date,
    payment_method      type_method                     not null,
    item_id             integer references item (id)    not null,
    service_id          integer references service (id) not null,
    master_id           integer references master (id)  not null
);

insert into city (name)
values ('Moscow'),
       ('Omsk'),
       ('Tomsk'),
       ('Saint-Petersburg'),
       ('Arkhangelsk'),
       ('Astrakhan'),
       ('Barnaul'),
       ('Krasnoyarsk'),
       ('Kemerovo'),
       ('Biysk'),
       ('Irkutsk');

insert into item(type, cost)
values ('fridge', '800'),
       ('vacuum_cleaner', '500'),
       ('fridge', '200'),
       ('vacuum_cleaner', '400'),
       ('coffee_maker', '2000'),
       ('vacuum_cleaner', '1000'),
       ('hairdryer', '700'),
       ('multicooker', '250'),
       ('fridge', '300'),
       ('vacuum_cleaner', '600');

insert into service(name, city_id, items)
values ('GSMaster', 1, '{fridge, vacuum_cleaner}'),
       ('AM Service', 2, '{fridge, vacuum_cleaner, coffee_maker}'),
       ('Спектрум-Сервис', 3, '{fridge, coffee_maker, hairdryer}'),
       ('Ремтехника', 4, '{fridge, vacuum_cleaner, coffee_maker, hairdryer}'),
       ('DNS Repair', 5, '{fridge, coffee_maker, hairdryer, microwave_oven}'),
       ('АтельеТехно', 6, '{fridge, vacuum_cleaner, coffee_maker, hairdryer, microwave_oven}'),
       ('PartsDirect', 7, '{fridge, vacuum_cleaner, coffee_maker, hairdryer, microwave_oven, multicooker}'),
       ('НоутЛенд', 8, '{fridge, vacuum_cleaner, coffee_maker, microwave_oven, multicooker}'),
       ('Техноврач', 9, '{vacuum_cleaner, hairdryer, microwave_oven, multicooker}'),
       ('Бирюса', 10, '{fridge, coffee_maker, hairdryer, microwave_oven}');

insert into master(name, surname, birthday, service_id, start_work, city_id)
values ('Ivan', 'Abramson', '1960-01-22', (select id from service order by random() limit 1), '2000-10-10', 1),
       ('James', ' Adrian', '1980-01-22', (select id from service order by random() limit 1), '2002-10-10', 2),
       ('John', 'Albertson', '1981-01-22', (select id from service order by random() limit 1), '2003-10-10', 3),
       ('Robert', 'Allford', '1980-05-22', (select id from service order by random() limit 1), '2004-10-10', 4),
       ('Michael', 'Arnold', '1990-01-22', (select id from service order by random() limit 1), '2005-10-10', 5),
       ('William', 'Attwood', '2000-11-23', (select id from service order by random() limit 1), '2020-10-10', 6),
       ('Richard', 'Backer', '1992-03-22', (select id from service order by random() limit 1), '2007-10-10', 7),
       ('Thomas', 'Barrington', '1986-04-02', (select id from service order by random() limit 1), '2020-10-10', 8),
       ('George', 'Bishop', '1980-01-05', (select id from service order by random() limit 1), '2000-10-10', 9),
       ('Kenneth', 'Bradberry', '1984-02-22', (select id from service order by random() limit 1), '2000-10-10', 10);

insert into repair_order (date_of_application, date_of_completion, payment_method, item_id, service_id, master_id)
values ('2022-12-06', '2022-12-22', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 1),
       ('2020-01-22', '2020-02-01', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 2),
       ('2020-01-23', '2020-02-06', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 3),
       ('2020-01-24', '2020-02-24', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 4),
       ('2020-01-25', '2020-02-15', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 5),
       ('2020-01-26', '2020-02-05', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 6),
       ('2020-01-27', '2020-03-01', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 7),
       ('2020-01-28', '2020-02-14', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 8),
       ('2020-01-29', '2020-03-05', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 9),
       ('2020-01-30', '2020-02-26', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1), 10);

-- Лабораторная №8
--Триггер выполняется перед удалением записи из таблицы мастеров.
--Триггер проверяет наличие в другой таблице записей, относящихся к удаляемому мастеру, и, если такие записи есть, удаляет их.
create function delete_master() returns trigger as
$$
begin
    delete from repair_order where master_id = old.id;
    return old;
end;
$$ language plpgsql;


create trigger delete_master
    before delete
    on master
    for each row
execute procedure delete_master();

--Триггер выполняется перед вставкой новой записи в таблицу мастеров.
-- Триггер проверяет значения, которые должна содержать новая запись и может их изменить:
-- eсли не указано имя мастера – оно генерируется по схеме – Master + уникальный номер из последовательности
-- eсли не указан город мастера– ставится значение по умолчанию – “Irkutsk”
-- если не указан стаж мастера или стаж <=5 – устанавливается стаж, равный 5 для мастера из города “Irkutsk” и 0 для всех остальных.
create or replace function insert_master()
    returns trigger
as
$$
declare
    city text;
begin
    select into city name from city where id = new.city_id;
    if new.name is null then
        new.name = 'Master' || nextval('master_id_seq');
    end if;
    if city is null then
        new.city_id = (select id from city where name = 'Irkutsk');
    end if;
    if new.start_work is null or
       justify_interval(((current_date - new.start_work) || ' days')::interval) <= '5 years'::interval then
        if city = 'Irkutsk' then
            new.start_work = current_date - '5 years'::interval;
        else
            new.start_work = current_date;
        end if;
    end if;
    return new;
end;
$$
    language plpgsql;

create trigger insert_master
    before insert
    on master
    for each row
execute procedure insert_master();

insert into master(name, surname, birthday, service_id, start_work, city_id)
values ('James', 'Bond', '1980-01-22', (select id from service order by random() limit 1), '2002-10-10',
        null), -- со стажем без города
       ('Ivan', 'Petrov', '1980-01-22', (select id from service order by random() limit 1), null,
        5),    -- без стажа не из иркутска
       ('Viktor', 'Ivanov', '1980-01-22', (select id from service order by random() limit 1), null,
        11); -- без стажа из иркутска

delete
from master
where id = 1;

create or replace function get_interval(date text) returns text as
$$
begin
    return justify_interval((date || ' days')::interval);
end;
$$
    language 'plpgsql';

create or replace function get_interval(date date) returns text as
$$
begin
    return justify_interval((current_date - date || ' days')::interval);
end;
$$
    language 'plpgsql';

select get_interval('2020'::text);
select get_interval('2020-01-01'::date);