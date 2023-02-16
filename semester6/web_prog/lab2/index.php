<html>

<head>
    <title>Лабораторная №2</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №2</h1>
    <h3>Вариант 17</h3>

    <?php

    session_start();

    ini_set('display_errors', '1');
    ini_set('display_startup_errors', '1');
    error_reporting(E_ALL);

    require_once 'config.php';

    $dbconn = pg_connect("host=$host port=$port user=$user password=$password dbname=$db")
        or die('Не удалось соединиться: ' . pg_last_error());


    $city_search = "";

    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["search"])) {
        if ($_POST["search"] != "")
            $city_search = ' where city.name = \'' . $_POST["search"] . '\'';
        else
            header("location:index.php");
    }

    if (isset($_POST["log_out"])) {
        session_destroy();
        header("location:index.php");
    }

    $authenticated = !empty($_SESSION['auth']);
    $authenticated = $authenticated && $_SESSION['auth'] == true;

    echo "<h3>Ваши права для базы данных: ";
    if ($authenticated && $_SESSION['group'] == 1) {
        echo "чтение и обновление данных";
    } else if ($authenticated && $_SESSION['group'] == 2) {
        echo "чтение, обновление данных и добавление новых записей";
    } else {
        echo "чтение данных и поиск";
    }
    echo "</h3>";

    $query = 'select ads.id, type.type, city.name as city, ads.address, ads.roominess, ads.price, ads.created_at from ads inner join ad_types as type on ads.type = type.id inner join cities as city on ads.city = city.id' . $city_search . ";";

    echo $query . "<br><br>";

    $result = pg_query($dbconn, $query) or die('Ошибка запроса: ' . pg_last_error());

    echo "<table border=\"1\">\n";
    echo "<tr>
    <th>id</th>
    <th>type</th>
    <th>city</th>
    <th>address</th>
    <th>roominess</th>
    <th>price</th>
    <th>date</th>";

    echo "<br><br>";
    echo var_dump($_SESSION);
    echo "<br><br>";

    if ($authenticated && $_SESSION['group'] >= 1) {
        echo "<th></th>";
        if ($authenticated && $_SESSION['group'] >= 2) {
            echo "<th></th>";
        }
    }

    echo "</tr>\n";

    while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
        echo "\t<tr>\n";
        foreach ($line as $col_value) {
            echo "\t\t<td>$col_value</td>\n";
        }
        if ($authenticated && $_SESSION['group'] >= 1) {
            echo "\t<td><a href=\"update.php?id=" . $line["id"] . "\">Update</a></td>";
            if ($authenticated && $_SESSION['group'] >= 2) {
                echo "\t<td><a href=\"delete.php?id=" . $line["id"] . "\">Delete</a></td>";
            }
        }
        echo "\t</tr>\n";
    }
    echo "</table>\n";

    // Очистка результата
    pg_free_result($result);

    // Закрытие соединения
    pg_close($dbconn);
    ?>
    <form method="post" action="index.php">
        <P><span>Поиск по городу: </span><br><input type="text" name="search"></P>
        <P><button type="submit">Поиск</button></P>
    </form>


    <?php

    if ($authenticated && $_SESSION['group'] == 2) {
        echo '<form method="post" action="insert.php">
        <p><button type="insert">Создать объявление</button></p></form>';
    }
    ?>

    <form method="get" action="login.php">
        <p><button type="submit" name="log_out">Вход</button></p>
    </form>

    <form method="post" action="index.php">
        <p><button type="submit" name="log_out">Выход</button></p>
    </form>




</body>

</html>