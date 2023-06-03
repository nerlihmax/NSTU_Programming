drop schema if exists public cascade;
create schema public;

create table departments
(
    id   serial primary key,
    name text not null
);

create table positions
(
    id   serial primary key,
    name text not null
);

create table employees
(
    id         serial primary key,
    name       text not null,
    surname    text not null,
    department int references departments (id) on delete cascade on update cascade,
    position   int references positions (id) on delete cascade on update cascade,
    hire_date  date not null
);

create table courses
(
    id          serial primary key,
    name        text not null,
    department  int references departments (id) on delete cascade on update cascade,
    hours       int  not null check ( hours > 0 ),
    description text
);

create table courses_completion
(
    id         serial primary key,
    employee   int references employees (id) on delete cascade on update cascade,
    course     int references courses (id) on delete cascade on update cascade,
    start_date date not null
);

insert into departments (name)
values ('IT'),
       ('HR'),
       ('Sales');

insert into positions (name)
values ('Manager'),
       ('Developer'),
       ('HR Manager'),
       ('Sales Manager');

insert into employees (name, surname, department, position, hire_date)
values ('John', 'Doe', 1, 2, '2010-01-01'),
       ('Jane', 'Doe', 1, 1, '2011-01-01'),
       ('Jack', 'Doe', 2, 3, '2012-01-01'),
       ('Jill', 'Doe', 3, 4, '2013-01-01');

insert into courses (name, department, hours, description)
values ('Java', 1, 30, 'Java course'),
       ('C#', 1, 30, 'C# course'),
       ('HR', 2, 30, 'HR course'),
       ('Sales', 3, 30, 'Sales course'),
       ('Python', 1, 30, 'Python course'),
       ('C++', 1, 30, 'C++ course');

select e.name, e.surname, c.name, cc.start_date
from employees e
         join courses_completion cc on e.id = cc.employee
         join courses c on cc.course = c.id
where extract(month from cc.start_date) = extract(month from now())
  and extract(year from cc.start_date) = extract(year from now());

--выдать справки о сотрудниках, проходящих обучение в данном месяце, годе;
insert into courses_completion (employee, course, start_date)
values (1, 1, '2023-01-01'),
       (1, 2, '2023-02-11'),
       (3, 3, '2023-03-16'),
       (3, 3, '2023-04-18'),
       (1, 5, '2023-05-25'),
       (1, 6, '2023-06-02'),
       (2, 1, '2023-07-09'),
       (2, 2, '2023-08-19'),
       (3, 3, '2023-09-21'),
       (4, 4, '2023-10-27'),
       (4, 4, '2022-06-15'),
       (2, 5, '2023-06-23'),
       (2, 6, '2023-12-14'),
       (4, 6, '2024-12-13');

--выдавать справки о сотрудниках, которые должны проходить обучение (срок больше года)
select e.name, e.surname, c.name, cc.start_date
from employees e
         join courses_completion cc on e.id = cc.employee
         join courses c on cc.course = c.id
where cc.start_date < now() - interval '1 year';

--выдавать справки о сотрудниках отделов с курсами, которые они прослушали
select e.name, e.surname, d.name, c.name, cc.start_date
from employees e
         join departments d on e.department = d.id
         join courses c on d.id = c.department
         join courses_completion cc on c.id = cc.course
where e.department = c.department;