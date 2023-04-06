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


<body class="flex flex-col mx-auto h-full w-full justify-top items-center py-10 space-y-10">
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
    <form method="post" action="auth.php"
        class="flex flex-col bg-white w-2/6 rounded-xl pt-4 justify-center items-center space-y-6 pb-4">
        <div class="w-72">
            <div class="relative h-10  w-full min-w-[200px]">
                <input name="login" required
                    class="peer h-full w-full rounded-[7px] border border-blue-gray-200 border-t-transparent bg-transparent px-3 py-2.5 font-sans text-sm font-normal text-blue-gray-700 outline outline-0 transition-all placeholder-shown:border placeholder-shown:border-blue-gray-200 placeholder-shown:border-t-blue-gray-200 focus:border-2 focus:border-amber-400 focus:border-t-transparent focus:outline-0 disabled:border-0 disabled:bg-blue-gray-50"
                    placeholder=" " />
                <label
                    class="before:content[' '] after:content[' '] pointer-events-none absolute left-0 -top-1.5 flex h-full w-full select-none text-[11px] font-normal leading-tight text-blue-gray-400 transition-all before:pointer-events-none before:mt-[6.5px] before:mr-1 before:box-border before:block before:h-1.5 before:w-2.5 before:rounded-tl-md before:border-t before:border-l before:border-blue-gray-200 before:transition-all after:pointer-events-none after:mt-[6.5px] after:ml-1 after:box-border after:block after:h-1.5 after:w-2.5 after:flex-grow after:rounded-tr-md after:border-t after:border-r after:border-blue-gray-200 after:transition-all peer-placeholder-shown:text-sm peer-placeholder-shown:leading-[3.75] peer-placeholder-shown:text-blue-gray-500 peer-placeholder-shown:before:border-transparent peer-placeholder-shown:after:border-transparent peer-focus:text-[11px] peer-focus:leading-tight peer-focus:text-amber-400 peer-focus:before:border-t-2 peer-focus:before:border-l-2 peer-focus:before:border-amber-400 peer-focus:after:border-t-2 peer-focus:after:border-r-2 peer-focus:after:border-amber-400 peer-disabled:text-transparent peer-disabled:before:border-transparent peer-disabled:after:border-transparent peer-disabled:peer-placeholder-shown:text-blue-gray-500">
                    Логин
                </label>
            </div>
        </div>

        <div class="w-72">
            <div class="relative h-10 w-full min-w-[200px]">
                <input name="password" required
                    class="peer h-full w-full rounded-[7px] border border-blue-gray-200 border-t-transparent bg-transparent px-3 py-2.5 font-sans text-sm font-normal text-blue-gray-700 outline outline-0 transition-all placeholder-shown:border placeholder-shown:border-blue-gray-200 placeholder-shown:border-t-blue-gray-200 focus:border-2 focus:border-amber-400 focus:border-t-transparent focus:outline-0 disabled:border-0 disabled:bg-blue-gray-50"
                    placeholder=" " />
                <label
                    class="before:content[' '] after:content[' '] pointer-events-none absolute left-0 -top-1.5 flex h-full w-full select-none text-[11px] font-normal leading-tight text-blue-gray-400 transition-all before:pointer-events-none before:mt-[6.5px] before:mr-1 before:box-border before:block before:h-1.5 before:w-2.5 before:rounded-tl-md before:border-t before:border-l before:border-blue-gray-200 before:transition-all after:pointer-events-none after:mt-[6.5px] after:ml-1 after:box-border after:block after:h-1.5 after:w-2.5 after:flex-grow after:rounded-tr-md after:border-t after:border-r after:border-blue-gray-200 after:transition-all peer-placeholder-shown:text-sm peer-placeholder-shown:leading-[3.75] peer-placeholder-shown:text-blue-gray-500 peer-placeholder-shown:before:border-transparent peer-placeholder-shown:after:border-transparent peer-focus:text-[11px] peer-focus:leading-tight peer-focus:text-amber-400 peer-focus:before:border-t-2 peer-focus:before:border-l-2 peer-focus:before:border-amber-400 peer-focus:after:border-t-2 peer-focus:after:border-r-2 peer-focus:after:border-amber-400 peer-disabled:text-transparent peer-disabled:before:border-transparent peer-disabled:after:border-transparent peer-disabled:peer-placeholder-shown:text-blue-gray-500">
                    Пароль
                </label>
            </div>
        </div>

        <button type=" submit" class="bg-amber-400 px-4 py-2 rounded">Войти</button>
    </form>
</body>