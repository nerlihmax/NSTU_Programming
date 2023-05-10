<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');
session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated || $_SESSION['group'] < 2) {
    header('location:forbidden.html');
    exit();
}

?>

<html>

<head>
    <title>Insertion</title>
    <style>
        * {
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            font-size: 16px;
            font-weight: 500;
        }

        body {
            display: block;
            text-align: -webkit-center;
        }

        .header {
            background-color: #333;
            color: #fff;
            padding: 4px;
            text-align: center;
        }

        .h1 {
            font-size: 30px;
            padding: 4px;
        }
    </style>
</head>

<body>
    <header class="header">
        <h1 class="h1">Вставка</h1>
    </header>
    <?php
    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["position"]) && isset($_POST["degree"]) && isset($_POST["course"]) && isset($_POST["surname"]) && isset($_POST["room"])) {
        $position_id = pg_fetch_all(pg_query_params($db, 'SELECT id from "position" where name = $1 limit 1;', array($_POST['position'])), PGSQL_ASSOC)[0]['id'];
        $degree_id = pg_fetch_all(pg_query_params($db, 'SELECT id from degree where name = $1 limit 1;', array($_POST['degree'])), PGSQL_ASSOC)[0]['id'];
        $course_id = pg_fetch_all(pg_query_params($db, 'SELECT id from courses where name = $1 limit 1;', array($_POST['course'])), PGSQL_ASSOC)[0]['id'];

        if (!$position_id || $position_id == 0) {
            pg_query_params($db, 'INSERT into "position" (name) values ($1);', array($_POST['position']));
            $position_id = pg_fetch_all(pg_query_params($db, 'select id from "position" where name = $1 limit 1;', array($_POST['position'])), PGSQL_ASSOC)[0];
        }

        if (!$degree_id || $degree_id == 0) {
            pg_query_params($db, 'INSERT into degree (name) values ($1);', array($_POST['degree']));
            $degree_id = pg_fetch_all(pg_query_params($db, 'select id from degree where name = $1 limit 1;', array($_POST['degree'])), PGSQL_ASSOC)[0];
        }

        if (!$course_id || $course_id == 0) {
            pg_query_params($db, 'INSERT into courses (name) values ($1);', array($_POST['course']));
            $course_id = pg_fetch_all(pg_query_params($db, 'select id from courses where name = $1 limit 1;', array($_POST['course'])), PGSQL_ASSOC)[0];
        }


        $result = pg_query_params(
            $db,
            'INSERT into teachers (position, degree, courses, surname, room_number) values ($1, $2, $3, $4, $5);',

            array(
                (int) $position_id,
                (int) $degree_id,
                (int) $course_id,
                $_POST['surname'],
                (int) $_POST['room']
            )
        );
        if ($result != false) {
            header("location:index.php#");
        }
    }

    ?>


    <form method="post" action="insert.php">
        <p><span>Должность: </span><input type="text" name="position" required></p>
        <p><span>Степень: </span><input type="text" name="degree" required></p>
        <p><span>Курс: </span><input type="text" name="course" required></p>
        <p><span>Фамилия: </span><input type="text" name="surname" required></p>
        <p><span>Кабинет: </span><input type="number" name="room" required></p>
        <p><button type="submit">Создать запись</button></p>
    </form>
</body>