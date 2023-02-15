<html>

<head>
    <title>Login</title>
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

    session_start();



    if (!empty($_POST['password']) and !empty($_POST['login'])) {
        $login = $_POST['login'];
        $password = $_POST['password'];

        $query = "SELECT * FROM users WHERE login=$1 AND password=$2";
        $result = pg_query_params($dbconn, $query, array($login, $password));
        $user = pg_fetch_assoc($result);

        if (!empty($user)) {
            $_SESSION['auth'] = true;
            $_SESSION['group'] = $user['access_level'];
            header("location:index.php");
        } else {
            echo 'Password or login incorrect <br><br>';
        }
    }

    ?>

    <form method="post" action="login.php">
        <p><span>Логин: </span><br><input type="text" name="login"></p>
        <p><span>Пароль: </span><br><input type="text" name="password"></p>
        <p><button type="submit">Войти в аккаунт</button></p>
    </form>
</body>