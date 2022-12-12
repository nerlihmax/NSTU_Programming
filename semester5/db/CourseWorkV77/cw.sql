drop schema if exists public cascade;
create schema if not exists public;

create type age_category as enum ('Для детей', 'Для подростков', 'Для взрослых', 'Для пенсионеров');

create table countries -- страны
(
    id   serial primary key,
    name text unique
);

create table manufacturers -- производители
(
    id      serial primary key,
    name    text unique,
    country int references countries (id)
);

create table providers -- поставщики
(
    id   serial primary key,
    name text unique
);

create table equipments -- спортинвентарь
(
    id             serial primary key,
    name           text not null,                                       -- наименование
    manufacturer   int references manufacturers (id) on delete cascade, -- производитель
    date_of_issue  date check ( date_of_issue < current_date ),         -- дата выпуска
    provider       int references providers (id) on delete cascade,     -- поставщик
    date_of_supply date check ( date_of_supply < current_date ),        -- дата поставки
    age_category   age_category,                                        -- возрастная категория
    price          int                                                  -- цена
);

create table equipment_rental -- прокат спортинвентаря
(
    id             serial primary key,
    equipment      int references equipments (id), -- ID спортивного товара
    start_of_lease date not null,                  -- дата начала проката
    end_of_lease   date                            -- дата окончания проката
);

insert into countries(name)
values ('Германия'),
       ('Бельгия'),
       ('Италия'),
       ('Франция'),
       ('США'),
       ('Малайзия'),
       ('Япония'),
       ('Нидерланды'),
       ('Чехия'),
       ('Великобритания');

insert into manufacturers(name, country)
values ('Adidas', 1),
       ('Anta Sports', 2),
       ('Babolat', 1),
       ('Blizzard Sport', 3),
       ('Bosco Sport', 4),
       ('Buka Boxing', 1),
       ('Burrda', 5),
       ('CAMP', 1),
       ('Daedo', 5),
       ('Diadora', 9),
       ('Dunlop Sport', 10),
       ('Errea', 10),
       ('Fila', 5),
       ('Fischer', 1),
       ('Helly Hansen', 3),
       ('Kappa', 6),
       ('Nordica', 10),
       ('Oakley', 10),
       ('Puma', 10),
       ('Rockport', 6),
       ('Umbro', 1);

insert into providers(name)
values ('ООО ОптусБай'),
       ('Спорткоробка'),
       ('Атлас Спорт'),
       ('ООО "РЕД СКИЛЛ"'),
       ('StreetAtlet'),
       ('RedfiveHockey'),
       ('Спортмастер'),
       ('Декатлон'),
       ('Takeshi Fight Gear'),
       ('ООО Хозлидер'),
       ('МК ОЛИМПСИТИ'),
       ('ТМ "PRIVAL"'),
       ('KMD'),
       ('BAYCON'),
       ('РЭЙ-СПОРТ'),
       ('Sport4Life');

create or replace function add_n_equipments(n integer) returns char --Функция для заполнения таблицы с товарами случайными данными
as
$$
declare
    t               int;
    _name           text;
    _manufacturer   int;
    _date_of_issue  date;
    _provider       int ;
    _date_of_supply date;
    _age_category   age_category;
    _price          int;
begin
    select max(id) into t from equipments;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            _name :=
                    (select (array [
                        'Воллейбольный мяч',
                        'Футбольный мяч',
                        'Ракетка для настольного тенниса',
                        'Ракетка для большого тенниса',
                        'Баскетбольный мяч',
                        'Атлетические булавы',
                        'Гантели',
                        'Горные лыжи',
                        'Беговые лыжи',
                        'Велосипед',
                        'Мячик для большого тенниса',
                        'Мячик для настольного тенниса',
                        'Штанга',
                        'Хоккейная клюшка',
                        'Хоккейная шайба',
                        'Лыжероллеры',
                        'Дартс',
                        'Комплект для бадминтона',
                        'Метательное ядро',
                        'Коврик для йоги',
                        'Шагомер',
                        'Эспандер'])[round(random() * 21 + 1)]::text);
            _date_of_issue := (select timestamp '2010-01-01 20:00:00' +
                                      random() * (timestamp '2020-12-31 20:00:00' -
                                                  timestamp '2016-01-10 20:00:00'));
            _manufacturer := (select id from manufacturers order by random() limit 1);
            _provider := (select id from providers order by random() limit 1);
            _date_of_supply := (_date_of_issue + random() * (timestamp '2020-12-31 20:00:00' -
                                                             timestamp '2016-01-10 20:00:00'));
            _price := (round(random() * 50000 + 200));
            _age_category :=
                    (select (array ['Для детей', 'Для подростков', 'Для взрослых', 'Для пенсионеров'])[round(random() * 3 + 1)]::age_category);
            insert into equipments (name, manufacturer, date_of_issue, provider, date_of_supply, age_category, price)
            values (_name, _manufacturer, _date_of_issue, _provider, _date_of_supply, _age_category, _price);
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

create or replace function add_n_equipment_rental(n integer) returns char --Функция для заполнения таблицы проката спортинвентаря случайными данными
as
$$
declare
    t               int;
    _equipment      int;
    _start_of_lease date;
    _end_of_lease   date;
begin
    select max(id) into t from equipments;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            _equipment := (select id from equipments order by random() limit 1);
            _start_of_lease := (select timestamp '2020-01-01 20:00:00' +
                                       random() * (timestamp '2022-12-31 20:00:00' -
                                                   timestamp '2021-01-01 20:00:00'));
            _end_of_lease := null;
            if (random() > 0.6) then
                _end_of_lease := (select _start_of_lease + justify_interval(random() * interval '2 year'));
            end if;
            insert into equipment_rental (equipment, start_of_lease, end_of_lease)
            values (_equipment, _start_of_lease, _end_of_lease);
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

select add_n_equipments(1000);
select add_n_equipment_rental(400);

drop role if exists operator;
drop role if exists db_user;
drop role if exists admin;

--Пересоздаем пользователей
create role operator with login password '12345';
create role db_user with login password '12345';
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
grant insert, select, update, delete on countries, manufacturers, providers to operator;
grant insert, select, update, delete on equipments, equipment_rental to db_user;
grant all privileges on all tables in schema public to admin;