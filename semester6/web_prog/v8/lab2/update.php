<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');
session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated && $_SESSION['group'] < 2) {
    header('location:forbidden.html');
    exit();
}
?>

<html>

<head>
    <title>Update</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <?php

    $queries = array();
    parse_str($_SERVER["QUERY_STRING"], $queries);

    $id = $queries["id"] or die("No id provided");

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
            'UPDATE teachers set (position, degree, courses, surname, room_number) = ($1, $2, $3, $4, $5) where id = $6;',

            array(
                (int) $position_id,
                (int) $degree_id,
                (int) $course_id,
                $_POST['surname'],
                (int) $_POST['room'],
                $id
            )
        );
        if ($result != false) {
            header("location:index.php#");
        }
    }

    $query =
        'SELECT teachers.id,
    position.name as position,
    degree.name as degree,
    courses.name as course,
    teachers.surname as teacher,
    teachers.room_number
    from teachers as teachers
    inner join position on position.id = teachers.position
    inner join degree on degree.id = teachers.degree
    inner join courses on courses.id = teachers.courses 
    where teachers.id = ' . $id;

    $result = pg_query($db, $query) or die('Ошибка запроса: ' . pg_last_error());

    if ($result == false) {
        echo "Failed to fetch row";
        header("location:index.php");
    }

    $row = pg_fetch_array($result, null, PGSQL_ASSOC);
    ?>

    <form method="post" action="update.php?id=<?php echo $id; ?>">
        <p><span>Должность: </span><input type="text" required name="position" value="<?= $row['position'] ?>"></p>
        <p><span>Степень: </span><input type="text" required name="degree" value="<?= $row['degree'] ?>"></p>
        <p><span>Курс: </span><input type="text" required name="course" value="<?= $row['course'] ?>"></p>
        <p><span>Фамилия: </span><input type="text" required name="surname" value="<?= $row['teacher'] ?>"></p>
        <p><span>Кабинет: </span><input type="number" required name="room" value="<?= $row['room_number'] ?>"></p>
        <p><button type="submit">Обновить</button></p>
    </form>
</body>