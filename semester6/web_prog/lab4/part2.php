<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <title>Часть 2</title>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.3/dist/jquery.min.js"></script>
    <script type="text /javascript" src="js/jquery-1.2.6.js"></script>
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h1>Вариант 17</h1>
    <h2>Часть 2</h2>

    <?php

    require_once 'config.php';

    ini_set('display_errors', '1');
    ini_set('display_startup_errors', '1');
    error_reporting(E_ALL);

    $dbconn = pg_connect('host=' . $host . ' port=' . $port . ' user=' . $user . ' password=' . $password . ' dbname=' . $db) or die('Не удалось соединиться: ' . pg_last_error());

    $query = 'select cities.name from cities;';

    $result = pg_query($dbconn, $query);

    $res_arr = array();


    while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
        array_push($res_arr, $line['name']);
    }
    ?>

    <h3>Выберите город</h3>
    <form class="form">
        <select name="city">>
            <?php
            foreach ($res_arr as $value) {
                echo "<option value=\"" . $value . "\">" . $value . "</option>";
            }
            ?>
        </select>
    </form>

    <br>

    <h3>Объявления</h3>

    <table class="list" border="1">
        <tr>
            <th>Тип</th>
            <th>Город</th>
            <th>Адрес</th>
            <th>Комнаты</th>
            <th>Цена</th>
            <th>Дата</th>
        <tr>
    </table>

    <style>
        td,
        th {
            padding: 4px;
            text-align: center;
        }
    </style>

    <script>
        const getAdsByCityName = async (cityName) => {
            const response = await fetch(`https://web.s.kheynov.ru/lab4/cities.php?city=${cityName}`);
            const json = await response.json();
            return json;
        };

        const renderList = ({ cities }) => {
            const list = document.querySelector('.list');

            [...list.children].forEach((child, idx) => idx > 0 && child.remove());

            cities
                .map((city) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${city.type}</td> 
                    <td>${city.city}</td> 
                    <td>${city.address}</td> 
                    <td>${city.roominess}</td> 
                    <td>${city.price}</td> 
                    <td>${city.created_at}</td>`;
                    return row;
                })
                .forEach((row) => list.appendChild(row));
        };

        const form = document.querySelector('.form');
        const { city } = form.elements;

        const app = async () => {
            const cities = await getAdsByCityName(city.value);
            renderList({ cities });
        };

        app();

        city.addEventListener('input', app);
    </script>

</body>

</html>