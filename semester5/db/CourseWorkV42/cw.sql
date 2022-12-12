drop schema if exists public cascade;
create schema if not exists public;

create table countries -- страны
(
    id   serial primary key,
    name text unique
);

create table manufacturers -- место производства (завод)
(
    id      serial primary key,
    name    text unique,
    country int references countries (id)
);

create table providers
(
    id      serial primary key,
    name    text unique,
    country int references countries (id)
);

create table photo_goods
(
    id             serial primary key,
    name           text                                          not null,
    date_of_issue  date                                          not null,
    date_of_supply date check ( date_of_supply > date_of_issue ) not null,
    provider       int references providers (id),
    manufacturer   int references manufacturers (id),
    price          int check ( price > 0 ),
    weight         int check ( weight > 0 ),
    date_of_sale   date check ( date_of_sale > date_of_supply ) default null
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
       ('Россия'),
       ('Великобритания');

insert into manufacturers(name, country)
values ('ADOX', 1),
       ('Agfa-Gevaert', 2),
       ('AgfaPhoto Holdings GMBH', 1),
       ('Беллини', 3),
       ('Bergger', 4),
       ('Calbe Chemie', 1),
       ('Carestream', 5),
       ('Compard', 1),
       ('Cinestill', 5),
       ('Foma Bohemia', 9),
       ('Fujifilm', 10),
       ('Harman Technology', 11),
       ('Kodak', 5),
       ('OPBO', 1),
       ('Yodica', 3),
       ('Champion', 6),
       ('Silberra', 10),
       ('Славич', 10),
       ('Олимп', 10),
       ('Тасма', 6),
       ('Zeiss', 1);

insert into providers(name, country)
values ('ALPA', 1),
       ('Технология Andor', 2),
       ('Angenieux', 3),
       ('apertus°', 4),
       ('Argus', 5),
       ('Арнольд и Рихтер', 6),
       ('Арри', 7),
       ('Asahi (см. Pentax)', 8),
       ('Bell & Howell Co.', 9),
       ('BELOMO', 10),
       ('Blackmagic Design', 3),
       ('Bolex', 1),
       ('Braun Nurnberg', 1),
       ('Bron Elektronik', 4),
       ('B +W Filterfabrik', 11),
       ('Cambo camera', 10),
       ('Кэнхэм', 11),
       ('Canon', 5),
       ('Casio', 1),
       ('Fujifilm', 7);

create or replace function add_n_photo_goods(n integer) returns char --Функция для заполнения таблицы с фототоварами рандомными значениями
as
$$
declare
    t                     int;
    declare _date_of_sale date;
begin
    select max(id) into t from photo_goods;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            if (random() > 0.4) then
                _date_of_sale := (select current_date - justify_interval(random() * interval '4 year'));
            end if;

            insert into photo_goods (name, date_of_issue, date_of_supply, provider, manufacturer, price, weight,
                                     date_of_sale)
            values ((select (array ['Фотобумага',
                'Светофильтр',
                'Бленда на объектив',
                'Кольцо переходное',
                'ИК-фильтр',
                'Наглазник',
                'Отражатель 60мм',
                'Кассета',
                'Фотоплёнка',
                'Чехол водонепроницаемый',
                'Бачок для проявки плёнки',
                'Синхронизатор вспышки',
                'Плёночный фотоаппарат',
                'Чехол для фотоаппарата',
                'Ремень для фотоаппарата',
                'Цифровой фотоаппарат',
                'Софтбокс'
                ])[round(random() * 16 + 1)]), --Выбираем случайное наименование
                    (select timestamp '2010-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')), -- рандомная дата производства
                    (select timestamp '2015-01-10 20:00:00' +
                            random() * (timestamp '2021-12-31 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')),
                    (select id from providers order by random() limit 1), -- рандомный поставщик
                    (select id from manufacturers order by random() limit 1), -- рандомный производитель
                    round(random() * 50000 + 1000), -- рандомная цена
                    round(random() * 900 + 100), -- рандомный вес в граммах
                    (_date_of_sale)); -- рандомная дата продажи
            _date_of_sale := null;
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

select add_n_photo_goods(600);


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
grant insert, select, update, delete on countries, manufacturers to operator;
grant insert, select, update, delete on providers, photo_goods to db_user;
grant all privileges on all tables in schema public to admin;
