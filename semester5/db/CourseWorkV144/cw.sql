drop schema if exists public cascade;
create schema if not exists public;

create type flowering_period as enum ('Winter', 'Spring', 'Summer', 'Autumn'); -- период цветения

create table cities --Таблица с городами
(
    id   serial primary key,
    name text unique
);

create table manufacturers --Производители
(
    id   serial primary key,
    name text unique,
    city integer references cities (id) not null
);

create table providers --Поставщики
(
    id   serial primary key,
    name text unique,
    city integer references cities (id) not null
);

create table diseases --Заболевания
(
    id   serial primary key,
    name text unique
);

create table medicinal_plants --Лекарственные растения
(
    id                  serial primary key,
    name                text,                                                 --Наименование
    manufacturer        integer references manufacturers (id)       not null, --Изготовитель
    provider            integer references providers (id)           not null, --Поставщик
    flowering_period    flowering_period                            not null, --Период цветения
    target_disease      integer references diseases (id)            not null, --Заболевание которое лечит
    price               integer check ( price > 0 )                 not null, --Цена
    date_of_issue       date check ( date_of_issue < current_date ) not null, --Дата производства
    date_of_supply      date,                                                 --Дата поставки
    sell_by             date                                        not null, --Срок годности
    storage_temperature integer                                     not null  --Температура хранения
);


create table sales --Таблица проданных растений
(
    id    serial primary key,
    plant integer references medicinal_plants (id), --идентификатор растения
    date  date check ( date < current_date )        -- Дата продажи
);

insert into cities (name)
values ('Нижний Новгород'),
       ('Алейск'),
       ('Астрахань'),
       ('Батайск'),
       ('Верещагино'),
       ('Вологда'),
       ('Воронеж'),
       ('Георгиевск'),
       ('Дальнегорск'),
       ('Ейск'),
       ('Железногорск'),
       ('Звенигород'),
       ('Ижевск'),
       ('Иркутск'),
       ('Камызяк'),
       ('Новосибирск');

insert into manufacturers (name, city)
values ('ООО «Агроберес»', 1),
       ('ООО «ЗДРАВА КРАСА»', 2),
       ('ООО «АНБ»', 3),
       ('ООО «ПК КОМФОРТ-СИБИРЬ»', 4),
       ('ООО «СЫРЬЕВАЯ КОМПАНИЯ СИБИРИ»', 5),
       ('ИП Шпакович Владимир Юрьевич', 6),
       ('ООО «ИВА»', 7),
       ('ООО «КОЛВИ-ЧЕЛЯБИНСК»', 8),
       ('ИП Апчаева Дарья Юрьевна', 9),
       ('ИП Городков Антон Сергеевич', 10),
       ('ООО «ТПК «АЛТАЙ-ЭКО»', 11),
       ('ООО «ШЕФ ТИ»', 12),
       ('ИП Абелов Марсель Магаруфович', 13),
       ('ООО «ОМЕГА»', 14),
       ('ИП Орлов Антон Владимирович', 15),
       ('ООО «КАИРУС»', 16);

insert into providers(name, city)
values ('Флавотека', 1),
       ('Древний травник', 2),
       ('Дары Алтая', 3),
       ('Таёжный дворик', 4),
       ('Лавка здоровья', 5),
       ('Алёшины сады', 6),
       ('Лес лекарь', 7),
       ('Вершки и корешки', 8),
       ('Живица', 9),
       ('Тайга craft', 10),
       ('Алтайская сказка', 11),
       ('Дикорос', 12),
       ('Алтай', 13),
       ('Золотой дракон', 14),
       ('Злат Алтая', 15),
       ('Алтай дар', 16);

insert into diseases (name)
values ('Дерматомикоз'),
       ('Диарея'),
       ('Мигрень'),
       ('Цистит'),
       ('Грипп'),
       ('Воспаление почек'),
       ('Тревожность'),
       ('Повышенное АД'),
       ('Сердечно-сосудистые заболевания'),
       ('Псориаз');

create or replace function add_n_products(n integer) returns char --Функция для заполнения таблицы с растениями рандомными значениями
as
$$
declare
    t int;
begin
    select max(id) into t from medicinal_plants;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            insert into medicinal_plants (name, manufacturer, provider, flowering_period, target_disease, price,
                                          date_of_issue, date_of_supply, sell_by, storage_temperature)
            values ((select (array [
                'Белокопытник',
                'Баранец пильчатый',
                'Барбарис обыкновенный',
                'Адонис пламенный',
                'Бетель',
                'Гинкго',
                'Горец земноводный',
                'Брюква',
                'Белладонна',
                'Берёза повислая',
                'Барвинок прямой',
                'Аденантера павлинья',
                'Борец белоустный',
                'Арника горная',
                'Алтей армянский',
                'Валериана лекарственная',
                'Витекс священный',
                'Айва',
                'Бедренец большой',
                'Белена чёрная',
                'Ваниль плосколистная'])[floor(random() * 21 + 1)]), --Выбираем случайное наименование
                    (select id from manufacturers order by random() limit 1), --Случайный производитель
                    (select id from providers order by random() limit 1), --Случайный поставщик
                    (select (array ['Winter', 'Summer', 'Spring', 'Autumn'])[floor(random() * 4 + 1)])::flowering_period,
                    (select id from diseases order by random() limit 1), --Случайная болезнь
                    (floor(random() * 10000 + 100)), --Случайная цена
                    (select timestamp '2020-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')), --Случайная дата производства
                    (select timestamp '2021-01-10 20:00:00' +
                            random() * (timestamp '2021-12-31 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')), --Случайная дата поставки
                    (select timestamp '2022-08-21 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')),
                    floor(random() * 18 + 10)); --Случайная температура хранения
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

create or replace function add_n_sales(n integer) returns char --Функция для заполнения таблицы с продажами случайными данными
as
$$
declare
    t int;
begin
    select max(id) into t from sales;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            insert into sales (plant, date)
            values ((select id from medicinal_plants order by random() limit 1),
                    (select timestamp '2020-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';


select add_n_products(500); --Вставляем 500 позиций растений
select add_n_sales(100); --100 продаж

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
grant insert, select, update, delete on manufacturers, providers, diseases, cities to operator;
grant insert, select, update, delete on medicinal_plants, sales to db_user;
grant all privileges on all tables in schema public to admin;