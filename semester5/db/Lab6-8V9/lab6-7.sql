drop schema if exists public cascade;
create schema public;

create type type_method as enum ('cash','cashless');
create type type_item as enum ('fridge', 'vacuum_cleaner', 'coffee_maker', 'hairdryer', 'multicooker', 'microwave_oven');
create type operation_type as enum ('Update', 'Insert', 'Delete');

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
    service_id          integer references service (id) not null
);


-- TRIGGERS
create table journal
(
    operation  operation_type not null,
    stamp      timestamp      not null,
    userid     text           not null,
    table_name text           not null,
    row_id     integer        not null
);

create or replace function log() returns trigger as
$$
begin

    if (tg_op = 'DELETE') then
        insert into journal select 'Delete', now(), user, tg_table_name, old.id - 1;
        return old;
    elsif (tg_op = 'UPDATE') then
        insert into journal select 'Update', now(), user, tg_table_name, new.id;
        return new;
    elsif (tg_op = 'INSERT') then
        insert into journal select 'Insert', now(), user, tg_table_name, new.id;
        return new;
    end if;
    return null;
end;
$$ language 'plpgsql';

create trigger city_log
    after insert or update or delete
    on city
    for each row
execute procedure log();
create trigger item_log
    after insert or update or delete
    on item
    for each row
execute procedure log();
create trigger master_log
    after insert or update or delete
    on master
    for each row
execute procedure log();
create trigger repair_order_log
    after insert or update or delete
    on repair_order
    for each row
execute procedure log();

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
       ('Biysk');

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

insert into repair_order (date_of_application, date_of_completion, payment_method, item_id, service_id)
values ('2022-12-06', '2022-12-22', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-22', '2020-02-01', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-23', '2020-02-06', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-24', '2020-02-24', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-25', '2020-02-15', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-26', '2020-02-05', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-27', '2020-03-01', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-28', '2020-02-14', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-29', '2020-03-05', 'cashless', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1)),
       ('2020-01-30', '2020-02-26', 'cash', (select id from item order by random() limit 1),
        (select id from service order by random() limit 1));

insert into master(name, surname, birthday, service_id, start_work)
values ('Ivan', 'Abramson', '1960-01-22', (select id from service order by random() limit 1), '2000-10-10'),
       ('James', ' Adrian', '1980-01-22', (select id from service order by random() limit 1), '2002-10-10'),
       ('John', 'Albertson', '1981-01-22', (select id from service order by random() limit 1), '2003-10-10'),
       ('Robert', 'Allford', '1980-05-22', (select id from service order by random() limit 1), '2004-10-10'),
       ('Michael', 'Arnold', '1990-01-22', (select id from service order by random() limit 1), '2005-10-10'),
       ('William', 'Attwood', '2000-11-23', (select id from service order by random() limit 1), '2020-10-10'),
       ('Richard', 'Backer', '1992-03-22', (select id from service order by random() limit 1), '2007-10-10'),
       ('Thomas', 'Barrington', '1986-04-02', (select id from service order by random() limit 1), '2020-10-10'),
       ('George', 'Bishop', '1980-01-05', (select id from service order by random() limit 1), '2000-10-10'),
       ('Kenneth', 'Bradberry', '1984-02-22', (select id from service order by random() limit 1), '2000-10-10');

-- create indices on tables
create index city_id_idx on city (name);
create index item_id_idx on item (type, cost);
create index master_id_idx on master (name, surname);
create index repair_order_id_idx on repair_order (service_id, payment_method);

--====USERS====
drop role if exists operator;
drop role if exists db_user;
drop role if exists admin;
drop role if exists analyst;

--Пересоздаем пользователей
create role operator with login password '12345';
create role db_user with login password '12345';
create role admin with login password '12345';
create role analyst with login password '12345';

--Выделяем права на схему public
grant usage on schema public to operator;
grant usage on schema public to db_user;
grant usage on schema public to admin;

grant all on schema public to admin;
grant all on schema public to operator;
grant all on schema public to db_user;

grant select, usage on all sequences in schema public to operator;
grant select, usage on all sequences in schema public to db_user;
grant select on all sequences in schema public to analyst;
grant all privileges on all sequences in schema public to admin;

grant select on all tables in schema public to analyst;
revoke all privileges on journal from analyst;

grant insert, select, update, delete on city, service to operator;
grant insert, select, update, delete on item, master, repair_order to db_user;
grant all privileges on all tables in schema public to admin;

grant insert on journal to operator, db_user;

--====FUNCTIONS====
--Зарегистрировать новый гарантийный случай
create or replace function create_repair_order(_date_of_application date, _date_of_completion date,
                                               _payment_method type_method, _item_id int, _service_id int)
    returns text as
$$
begin
    if _date_of_application > _date_of_completion then
        raise exception 'date_of_application > date_of_completion';
    end if;
    if _payment_method != 'cashless' and _payment_method != 'cash' then
        raise exception 'payment_method != cashless or cash';
    end if;

    insert into repair_order (date_of_application, date_of_completion, payment_method, item_id, service_id)
    values (_date_of_application, _date_of_completion, _payment_method, _item_id, _service_id);
    return 'OK';
end;
$$ language plpgsql;

select create_repair_order('2022-12-15', '2022-12-31', 'cash', 1, 1);


--Зарегистрировать нового мастера
create or replace function create_master(_name varchar(20), _surname varchar(20), _birthday date, _service_id int,
                                         _start_work date)
    returns text as
$$
begin
    if _start_work > now() then
        raise exception 'start_work > now()';
    end if;
    if (select id from service where id = _service_id) is null then
        raise exception 'service not found';
    end if;
    insert into master (name, surname, birthday, service_id, start_work)
    values (_name, _surname, _birthday, _service_id, _start_work);
    return 'OK';
end;
$$ language plpgsql;

select create_master('Ivan', 'Ivanov', '1980-01-20', 1, '2020-01-20');

--Зарегистрировать новый предмет ремонта
create or replace function create_item(_type type_item, _cost int)
    returns text as
$$
begin
    if _cost < 0 then
        raise exception 'cost < 0';
    end if;
    insert into item (type, cost)
    values (_type, _cost);
    return 'OK';
end;
$$ language plpgsql;

select create_item('fridge', 1000);

--Зарегистрировать новую станцию гарантийного обслуживания
create or replace function create_service(_name text, _city_id int, _items type_item[])
    returns text as
$$
begin
    insert into service (name, city_id, items)
    values (_name, _city_id, _items);
    return 'OK';
end;
$$ language plpgsql;


--====ANALYTICS====
--среднее время ожидания окончания ремонта предмета
create or replace function get_item_avg_repair_time()
    returns table
            (
                item_type type_item,
                avg_time  interval
            )
as
$$
begin
    return query
        select item.type ::type_item,
               justify_interval((avg(repair_order.date_of_completion - repair_order.date_of_application) ||
                                 ' days')::interval)
        from repair_order
                 join item on repair_order.item_id = item.id
        group by item.type;
end;
$$
    language plpgsql;

select *
from get_item_avg_repair_time();

--анализ объемов работ (по гарантийным мастерским)
create or replace function get_service_workload()
    returns table
            (
                service_name text,
                workload     real
            )
as
$$
begin
    return query
        select s.name ::text,
               count(repair_order.id)::real as workload
        from repair_order
                 inner join service s on repair_order.service_id = s.id
        group by s.name
        order by workload desc;
end;
$$
    language plpgsql;

select *
from get_service_workload();

--стоимость ремонта (средняя, максимальная, минимальная) в зависимости от города.
create or replace function get_repair_cost()
    returns table
            (
                city_name text,
                avg_cost  real,
                max_cost  real,
                min_cost  real
            )
as
$$
begin
    return query
        select c.name::text,
               avg(item.cost)::real,
               max(item.cost)::real,
               min(item.cost)::real
        from repair_order
                 join item on repair_order.item_id = item.id
                 join service s on repair_order.service_id = s.id
                 join city c on s.city_id = c.id
        group by c.name;
end;
$$
    language plpgsql;

select *
from get_repair_cost();

--рейтинг убыточности предметов для ремонта
--Прибыльность: стоимость выполнения / время выполнения, чем больше, тем прибыльнее ремонт предмета
create or replace function get_item_profitability()
    returns table
            (
                item_type     type_item,
                profitability integer
            )
as
$$
begin
    return query
        select item.type ::type_item,
               round(avg(item.cost::double precision /
                         (ro.date_of_completion - ro.date_of_application)::double precision))::int as profitability
        from repair_order ro
                 join item on ro.item_id = item.id
        group by item.type
        order by profitability desc;
end;
$$
    language plpgsql;
select *
from get_item_profitability();

--обеспеченность каждого города сервисами по разным группам предметов
create or replace function get_city_service_coverage()
    returns table
            (
                city_name text,
                item_type type_item[]
            )
as
$$
begin
    return query
        select c.name ::text,
               service.items
        from service
                 inner join master on service.id = master.service_id
                 inner join item it on it.type = any (service.items)
                 inner join city c on c.id = service.city_id
        group by c.name, service.items
        order by c.name;
end;
$$
    language plpgsql;

select *
from get_city_service_coverage();

--обеспеченность каждого города мастерскими по всему ассортименту в целом
create or replace function get_city_service_coverage_all()
    returns table
            (
                city_name text,
                coverage  int
            )
as
$$
begin
    return query
        with types_count as (select count(distinct item.type) from item)
        select c.name::text,
               round(count(distinct it.type)::double precision / (select * from types_count)::double precision * 100)::int
        from service
                 inner join master on service.id = master.service_id
                 inner join item it on it.type = any (service.items)
                 inner join city c on c.id = service.city_id
        group by c.name;
end;
$$
    language plpgsql;

select city_name, coverage || '%' as coverage
from get_city_service_coverage_all();