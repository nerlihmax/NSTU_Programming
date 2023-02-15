<html>

<head>
    <title>Row insertion</title>
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

    $dbconn = pg_connect("host=$host port=$port user=$user password=$password dbname=$db")
        or die('Не удалось соединиться: ' . pg_last_error());

    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["type"]) && isset($_POST["city"]) && isset($_POST["address"]) && isset($_POST["roominess"]) && isset($_POST["price"])) {
        $types = array_map('trim', pg_copy_to($dbconn, '(select type from ad_types)'));
        $type_id = array_search($_POST["type"], $types) + 1;

        $cities = array_map('trim', pg_copy_to($dbconn, '(select name from cities)'));
        $city_id = array_search($_POST["city"], $cities) + 1;
        $result = pg_query_params(
            $dbconn,
            'insert into ads(type, city, address, roominess, price) values ($1, $2, $3, $4, $5);',
            array(
                $type_id,
                $city_id,
                $_POST["address"],
                (int) $_POST["roominess"],
                (int) $_POST["price"]
            )
        );
        if ($result != false)
            header("location:index.php");
    }

    ?>



    <form method="post" action="insert.php">
        <select name="city">>
            <option value="Санкт-Петербург">Санкт-Петербург</option>
            <option value="Москва">Москва</option>
            <option value="Краснодар">Краснодар</option>
            <option value="Владивосток">Владивосток</option>
            <option value="Новосибирск">Новосибирск</option>
        </select>
        <select name="type">>
            <option value="Сдам">Сдам</option>
            <option value="Сниму">Сниму</option>
            <option value="Куплю">Куплю</option>
            <option value="Продам">Продам</option>
        </select>
        <p><span>Адрес: </span><textarea type="text" name="address"></textarea></p>
        <p><span>Количество комнат: </span><select name="roominess">>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
            </select>
        </p>
        <p><span>Цена: </span><input type="text" name="price"></p>
        <p><button type="submit">Создать объявление</button></p>
    </form>
</body>