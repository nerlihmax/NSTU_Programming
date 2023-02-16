<html>

<head>
    <title>Web programming NSTU course</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №3</h1>
    <h3>Вариант 17</h3>
    <h3>Часть 1</h3>

    <?php
    ini_set('display_errors', '1');
    ini_set('display_startup_errors', '1');
    error_reporting(E_ALL);

    require_once 'config.php';

    $dbconn = pg_connect('host=' . $host . ' port=' . $port . ' user=' . $user . ' password=' . $password . ' dbname=' . $db)
        or die('Не удалось соединиться: ' . pg_last_error());

    $query = 'select ads.id, type.type, city.name as city, ads.address, ads.roominess, ads.price, ads.created_at from ads inner join ad_types as type on ads.type = type.id inner join cities as city on ads.city = city.id;';

    $result = pg_query($dbconn, $query) or die('Ошибка запроса: ' . pg_last_error());

    echo "<table border=\"1\">\n";
    echo "<tr>
    <th>id</th>
    <th>type</th>
    <th>city</th>
    <th>address</th>
    <th>roominess</th>
    <th>price</th>
    <th>date</th>
    </tr>\n";

    while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
        echo "\t<tr>\n";
        foreach ($line as $col_value) {
            echo "\t\t<td>$col_value</td>\n";
        }
        echo "\t</tr>\n";
    }
    echo "</table>\n";

    // Очистка результата
    pg_free_result($result);

    // Закрытие соединения
    pg_close($dbconn);
    ?>

    <br><br>

    <h3>Количество объявлений/Количество комнат в квартирах</h3>

    <style>
        #v-legend {
            display: inline-block;
            transform: rotateZ(-90deg);
            position: absolute;
            top: 50%;
            left: -20%;
        }

        .graph {
            position: relative;
            max-width: fit-content;
        }
    </style>

    <div class="graph">
        <span id="v-legend">Количество объявлений</span>
        <img id="image" src="charts.php" alt="">
    </div>

    <span>Количество комнат в квартире</span>

    <script>
        setInterval(() => {
            image.src = 'charts.php?' + new Date().getTime();
        }, 1000);
    </script>
</body>

</html>