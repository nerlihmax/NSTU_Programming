<html>

<head>
    <title>Row update</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>

    <?php
    require_once 'config.php';

    ini_set('display_errors', '1');
    ini_set('display_startup_errors', '1');
    error_reporting(E_ALL);

    session_start();

    $authenticated = !empty($_SESSION['auth']);
    $authenticated = $authenticated && $_SESSION['auth'] == true;

    if (!$authenticated && $_SESSION['group'] < 1) {
        header('location:forbidden.html');
        exit();
    }

    $queries = array();
    parse_str($_SERVER["QUERY_STRING"], $queries);


    $dbconn = pg_connect("host=$host port=$port user=$user password=$password dbname=$db")
        or die('Не удалось соединиться: ' . pg_last_error());

    $id = $queries["id"] or die("No id provided");

    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["city"]) && isset($_POST["address"]) && isset($_POST["roominess"]) && isset($_POST["price"])) {
        $city_query = pg_query($dbconn, 'select id from cities where name = \'' . $_POST["city"] . '\';');
        $city = pg_fetch_array($city_query, null, PGSQL_ASSOC)["id"];
        $result = pg_query_params(
            $dbconn,
            'update ads set (city, address, roominess, price) = ($2, $3, $4, $5) where id = $1',
            array(
                $id,
                $city, $_POST["address"], $_POST["roominess"], $_POST["price"]
            )
        );
        if ($result != false)
            header("location:index.php");
    }


    $query = 'select ads.id, type.type, city.name as city, ads.address, ads.roominess, ads.price, ads.created_at from ads inner join ad_types as type on ads.type = type.id inner join cities as city on ads.city = city.id where ads.id = ' . $id;

    $result = pg_query($dbconn, $query) or die('Ошибка запроса: ' . pg_last_error());

    if ($result == false) {
        echo "Failed to fetch row";
        header("location:index.php");
    }

    $row = pg_fetch_array($result, null, PGSQL_ASSOC);


    ?>



    <form method="post" action="update.php?id=<?php echo $id; ?>">
        <select name="city">>
            <option value="Санкт-Петербург" <?php if ($row["city"] == "Санкт-Петербург")
                echo "selected"; ?>>
                Санкт-Петербург</option>
            <option value="Москва" <?php if ($row["city"] == "Москва")
                echo "selected"; ?>>Москва</option>
            <option value="Краснодар" <?php if ($row["city"] == "Краснодар")
                echo "selected"; ?>>Краснодар</option>
            <option value="Владивосток" <?php if ($row["city"] == "Владивосток")
                echo "selected"; ?>>Владивосток</option>
            <option value="Новосибирск" <?php if ($row["city"] == "Новосибирск")
                echo "selected"; ?>>Новосибирск</option>
        </select>

        <p><span>Адрес: </span><textarea type="text" name="address"><?php echo $row["address"] ?></textarea></p>
        <p><span>Количество комнат: </span><input type="text" name="roominess" value=<?php echo $row["roominess"] ?>>
        </p>
        <p><span>Цена: </span><input type="text" name="price" value=<?php echo $row["price"] ?>></p>
        <p><button type="submit">Поиск</button></p>
    </form>
</body>