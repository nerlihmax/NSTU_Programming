drop schema if exists public cascade;
create schema if not exists public;

create type vessel_type as enum ('Моторная яхта', 'Парусная яхта', 'Парусно-моторная яхта', 'Гулет', 'Теплоход', 'Скоростной катер');

create type vessel_status as enum ('Ремонтируется', 'Подлежит ремонту', 'Готова к эксплуатации');

create table countries -- страны
(
    id   serial primary key,
    name text unique
);

create table factories -- место производства (завод)
(
    id      serial primary key,
    name    text unique,
    country int references countries (id)
);

create table boats
(
    id            serial primary key,
    vessel_type   vessel_type not null,                          -- тип судна
    date_of_issue date check ( date_of_issue < current_date ),   -- дата производства
    lifetime      interval,                                      -- срок эксплуатации
    factory       int references factories (id),                 -- завод изготовитель (место производства)
    price         int check ( price > 0 ),                       -- Цена
    max_speed     int,                                           -- скорость
    country       int references countries (id),                 -- страна поставки
    status        vessel_status default 'Готова к эксплуатации', -- статус
    roominess     int check ( roominess > 0 ),                   -- вместительность
    date_of_sale  date          default null                     -- дата продажи
);



create table replacement_part_providers -- поставщики запчастей
(
    id      serial primary key,
    name    text,                         -- наименование
    country int references countries (id) -- страна поставщика
);

create table replacement_parts -- запасные части
(
    id          serial primary key,
    name        text,                                           -- наименование
    provider    int references replacement_part_providers (id), -- поставщик
    vessel_type vessel_type,                                    -- тип судна, для которого предназначена запчасть
    price       int check ( price > 0 )                         -- цена
);

insert into countries (name)
values ('Италия'),
       ('США'),
       ('Нидерланды'),
       ('Германия'),
       ('Тайвань');

insert into factories (name, country)
values ('LVMH', 1),
       ('Gavio Group', 2),
       ('Privinvest Group', 3),
       ('Azimut-Benetti Group', 4),
       ('Ferretti Group', 5),
       ('Pershing', 1),
       ('Itama', 2),
       ('Heesen Yachts', 3),
       ('Oceanco', 4),
       ('Sunseeker', 5);

insert into replacement_part_providers (name, country)
values ('Vetus', 1),
       ('Jabsco', 2),
       ('Lewmar', 3),
       ('Webasto', 4),
       ('Hella marine', 5),
       ('Kobelt', 1),
       ('VDO', 2),
       ('Tecma', 3),
       ('Whale', 4),
       ('Aquadrive', 5),
       ('Barigo', 2),
       ('Schenker', 1),
       ('Dometic', 3),
       ('Minox', 2),
       ('Baratta', 4),
       ('Loipart', 5),
       ('Stork', 3);

create or replace function add_n_replacement_parts(n integer) returns char --Функция для заполнения таблицы с растениями рандомными значениями
as
$$
declare
    t int;
begin
    select max(id) into t from replacement_parts;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            insert into replacement_parts (name, provider, vessel_type, price)
            values ((select (array ['Якорь',
                'Радар',
                'Эхолот',
                'Компас',
                'Картплоттер',
                'Сигнальное средство',
                'Обогреватель',
                'Навигационные огни',
                'Стеклоочистители',
                'Привальные брусы',
                'Кранцы',
                'Тенты',
                'Чехлы',
                'Палубное сиденье',
                'Лестница',
                'Крыльчатка турбины',
                'Штурвал'
                ])[round(random() * 16 + 1)]), --Выбираем случайное наименование
                    (select id from replacement_part_providers order by random() limit 1),
                    (select (array ['Моторная яхта', 'Парусная яхта', 'Парусно-моторная яхта', 'Гулет', 'Теплоход', 'Скоростной катер'])[round(random() * 5 + 1)]::vessel_type),
                    round(random() * 50000 + 1000));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

select add_n_replacement_parts(100);

create or replace function add_n_boats(n integer) returns char --Функция для заполнения таблицы с продажами случайными данными
as
$$
declare
    t              int;
    _vessel_type   vessel_type;
    _date_of_issue date;
    _lifetime      interval;
    _factory       int;
    _price         int;
    _max_speed     int;
    _country       int;
    _status        vessel_status;
    _roominess     int;
    _date_of_sale  date;
begin
    select max(id) into t from boats;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            _vessel_type :=
                    (select (array [
                        'Моторная яхта',
                        'Парусная яхта',
                        'Парусно-моторная яхта',
                        'Гулет',
                        'Теплоход',
                        'Скоростной катер'])[round(random() * 5 + 1)]::vessel_type);
            _date_of_issue := (select timestamp '2010-01-01 20:00:00' +
                                      random() * (timestamp '2022-08-20 20:00:00' -
                                                  timestamp '2016-01-10 20:00:00'));

            _date_of_sale := null;
            _lifetime := null;
            if (random() > 0.4) then
                _date_of_sale := (select current_date - justify_interval(random() * interval '6 year'));
                _lifetime := justify_interval(((current_date - _date_of_sale) || ' days')::interval);
            end if;
            _factory := (select id from factories order by random() limit 1);
            _price := (round(random() * 50000000 + 2000000));
            _max_speed := (round(random() * 300 + 100));
            _country := (select id from countries order by random() limit 1);
            _status := (select (array [
                'Ремонтируется', 'Подлежит ремонту', 'Готова к эксплуатации'])[round(random() * 2 + 1)]::vessel_status);
            _roominess := round(random() * 10 + 1);
            insert into boats (vessel_type, date_of_issue, lifetime, factory, price, max_speed, country, status,
                               roominess, date_of_sale)
            values (_vessel_type, _date_of_issue, _lifetime, _factory, _price, _max_speed, _country, _status,
                    _roominess, _date_of_sale);
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';


select add_n_boats(400);


drop role if exists operator;
drop role if exists db_user;

--Если нет пользователя admin -- убрать следующую строку
drop role if exists admin;

--Пересоздаем пользователей
create role operator with login password '12345';
create role db_user with login password '12345';

--Если нет пользователя admin -- убрать следующую строку
create role admin with login password '12345';

--Выделяем права на схему public
grant usage on schema public to operator;
grant usage on schema public to db_user;
grant usage on schema public to admin;

grant all on schema public to admin;
grant all on schema public to operator;
grant all on schema public to db_user;


grant select, usage on all sequences in schema public to operator;
grant select, usage on all sequences in schema public to db_user;
grant all privileges on all sequences in schema public to admin;


--Выдаем права на таблицы справочники оператору, на остальные таблицы обычному пользователю, администратору на все
grant insert, select, update, delete on countries, replacement_part_providers, factories to operator;
grant insert, select, update, delete on replacement_parts, boats to db_user;
grant all privileges on all tables in schema public to admin;