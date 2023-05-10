<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v31');

session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated && $_SESSION['group'] < 2) {
    header('location:no_access.html');
    exit();
}

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$id = $queries["id"] or die("No id provided");

$operations = pg_fetch_all(pg_query($db, 'SELECT id, name from operation;'));

if (
    $_SERVER['REQUEST_METHOD'] == 'POST'
    && isset($_POST["operation"])
    && isset($_POST["name"])
    && isset($_POST["duration"])
) {
    $result = pg_query_params(
        $db,
        'UPDATE technological_map set
            name = $1, operation = $2, duration = $3 where id = $4;',
        array(
            $_POST['name'],
            (int) $_POST['operation'],
            (int) $_POST['duration'],
            $id,
        )
    );
    if ($result != false) {
        header("location:index.php#");
    }

}

?>

<html>

<head>
    <title>Изменение данных</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <?php

    $query = 'SELECT 
        map.id, 
        operation.id as operation_id, 
        map.name as detail, 
        map.duration
    from technological_map as map
             inner join operation on map.operation = operation.id
    WHERE map.id = $1';

    $result = pg_query_params($db, $query, array($id)) or die('Ошибка запроса: ' . pg_last_error());

    if ($result == false) {
        die("Failed to fetch row");
    }

    $row = pg_fetch_array($result, null, PGSQL_ASSOC);
    ?>

    <form method="post" action="update.php?id=<?php echo $id; ?>">
        <label for="operation">Вид операции</label>
        <select id="operation" name="operation">
            <?php
            foreach ($operations as &$operation) {
                ?>
                <option <?= ($operation["id"] == $row["operation_id"]) ? 'selected' : '' ?>
                    value=<?= htmlspecialchars($operation["id"]) ?>>
                    <?= htmlspecialchars($operation["name"]) ?>
                    <?php
            }
            ?>
        </select>
        <br><br>
        <label for="name">Название детали</label>
        <input id="name" name="name" type="text" value="<?= $row["detail"] ?>" />
        <br><br>

        <label for="duration">Длительность обработки (минуты): </label>
        <input type="number" name="duration" id="duration" min=10 value="<?= $row["duration"] ?>">
        <br><br>
        <p>
            <button type="submit">Редактировать запись</button>
        </p>
    </form>
</body>