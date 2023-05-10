<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v12');
?>

<html>

<head>
    <title>Login Page</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>


<body>
    <?php
    session_start();

    if (!empty($_POST['password']) and !empty($_POST['login'])) {
        $login = $_POST['login'];
        $password = $_POST['password'];

        $query = "SELECT * FROM users WHERE login=$1 AND password=$2";
        $result = pg_query_params($db, $query, array($login, $password));
        $user = pg_fetch_assoc($result);

        if (!empty($user)) {
            $_SESSION['auth'] = true;
            $_SESSION['group'] = $user['access_level'];
            $_SESSION['login'] = $user['login'];
            header("location:index.php");
        } else {
            echo 'Пароль или логин неправильный <br><br>';
        }
    }

    ?>
    <form method="post" action="login.php">
        <input id="login" name="login" required />
        <label for="login">Логин</label>
        <br>
        <br>
        <input id="password" name="password" required />
        <label for="password">Пароль</label>
        <br>
        <br>
        <button type=" submit">Войти</button>
    </form>
</body>