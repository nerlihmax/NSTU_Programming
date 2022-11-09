drop schema if exists public cascade;
create schema if not exists public;

drop role if exists operator;
drop role if exists db_user;

create table countries
(
    id           serial primary key,
    country_code char(2) unique not null,
    name         text
);

create table providers
(
    id         serial primary key,
    name       text,
    country_id integer references countries (id)
);

create table products
(
    id             serial primary key,
    name           text,
    date_of_issue  date,
    country_id     integer references countries (id),
    date_of_supply date,
    is_defect      boolean,
    provider_id    integer references providers (id),
    price          integer
);


create table sales
(
    id           serial primary key,
    product_id   integer references products (id),
    quantity     integer,
    date_of_sale date
);

create role operator with login password 'qwerty';
create role db_user with login password 'qwerty';

grant usage on schema public to operator;
grant usage on schema public to db_user;
grant usage on schema public to admin;

grant all on schema public to admin;
grant all on schema public to operator;
grant all on schema public to db_user;

grant select, usage on all sequences in schema public to operator;
grant select, usage on all sequences in schema public to db_user;
grant all privileges on all sequences in schema public to admin;

grant insert, select, update, delete on countries, providers to operator;
grant insert, select, update, delete on products, sales to db_user;
grant all privileges on all tables in schema public to admin;

insert into countries (country_code, name)
values ('RU', 'Russia'),
       ('JP', 'Japan'),
       ('GB', 'United Kingdom'),
       ('CN', 'China'),
       ('US', 'United States'),
       ('GE', 'Germany'),
       ('KR', 'South Korea'),
       ('UZ', 'Uzbekistan'),
       ('FR', 'France'),
       ('IT', 'Italy'),
       ('BZ', 'Brazil'),
       ('CA', 'Canada');
insert into providers (name, country_id)
values ('ООО «Рыбные Приколы»', 1),
       ('ОАО «Рыбак&Рыбачок»', 2),
       ('ЗАО «Мы продаём рыбов»', 3),
       ('АО «Рыбалка оптом»', 4),
       ('ЧП «рыбалка-снасти.рф»', 5),
       ('ООО «НИШТЯК»', 6),
       ('ИП «Рыбак Александр Игоревич»', 7),
       ('ООО «Сплыв»', 8),
       ('ОАО «Fishbaza»', 9),
       ('ООО «Дело Водяное»', 10),
       ('ООО «Рай для рыбака»', 11),
       ('ООО «Рыболов-эксперт»', 2),
       ('ООО «Карагем»', 6),
       ('ООО «База»', 3),
       ('ООО «Азимут»', 1),
       ('ООО «Легион Фиш»', 4),
       ('ООО «Фабрика54»', 2),
       ('ООО «Рыбалка круглый год»', 5),
       ('ООО «Сибирский медведь»', 9),
       ('ООО «Сибпромодежда»', 8);

create or replace function add_n_products(n integer) returns char
as
$$
declare
    t int;
begin
    select max(id) into t from products;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            insert into products (name, date_of_issue, country_id, date_of_supply, is_defect, provider_id,
                                  price)
            values ((select (array ['Балда', 'Вентерь','Удочка', 'Гарпун', 'Грузило', 'Катушка', 'Леска', 'Крючок', 'Спиннинг', 'Закидушка', 'Поплавок'])[floor(random() * 11 + 1)]),
                    (select timestamp '2010-01-10 20:00:00' +
                            random() * (timestamp '2019-01-20 20:00:00' -
                                        timestamp '2010-01-10 20:00:00')),
                    (select floor(random() * 12 + 1)),
                    (select timestamp '2020-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')),
                    (select random() > 0.97),
                    (select floor(random() * 20 + 1)),
                    (select floor(random() * 100000 + 200)));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

create or replace function add_n_sales(n integer) returns char
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
            insert into sales (product_id, quantity, date_of_sale)
            values ((select id from products where id > floor(random() * 10000 + 1) and is_defect = false limit 1),
                    (select floor(random() * 10 + 1)),
                    (select timestamp '2020-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

select add_n_products(10000);
select add_n_sales(200);