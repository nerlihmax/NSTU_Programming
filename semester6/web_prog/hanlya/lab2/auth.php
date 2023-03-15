<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();
?>

<html>

<head>
    <title>Authentication</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>


<body class="flex flex-col mx-auto h-full w-full justify-center items-center bg-gray-400 py-10 space-y-10">
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
            echo 'Password or login incorrect <br><br>';
        }
    }

    ?>
    <form method="post" action="auth.php"
        class="flex flex-col bg-white w-2/6 rounded-xl pt-10 justify-center items-center space-y-3 pb-4">
        <div><span>Логин: </span><input id="login" type="text" name="login"
                class="w-full h-8 rounded select-none px-4 py-2 bg-gray-200">
        </div>
        <div><span>Пароль: </span><input id="password" type="text" name="password"
                class="w-full h-8 rounded select-none px-4 py-2 bg-gray-200">
        </div>
        <button type=" submit" class="bg-amber-400 px-4 py-2 rounded">Войти</button>
    </form>
</body>