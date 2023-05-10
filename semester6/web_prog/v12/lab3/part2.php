<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

include_once("fckeditor/fckeditor.php");
session_start();
?>


<html>

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Лабораторная №3 часть 2</title>
</head>

<body>
    <form action="journal.php" method="post" target="_self">
        <h3>Комментарий</h3>
        <div id="ckeditor">
            <?php
            $oFCKeditor = new FCKeditor('text_input');
            $oFCKeditor->BasePath = './fckeditor/';
            $oFCKeditor->Value = '';
            $oFCKeditor->Create();
            ?>
        </div>
        <br>

        <h3>Выберите вид приветствия: </h3>
        <div>
            <input type="checkbox" name="styles[]" value="official" /> Официальное<br />
            <input type="checkbox" name="styles[]" value="nonformal" /> Неофициальное<br />
            <input type="checkbox" name="styles[]" value="basic" /> Простое<br />
        </div>

        <h3>Выберите размер: </h3>
        <select name="size">
            <option value="10 pt" selected="selected">Маленький</option>
            <option value="18 pt">Средний</option>
            <option value="32 pt">Большой</option>
        </select>

        <br>
        <h3>Капча</h3>
        <img src="/lab3/kcaptcha/index.php?<?php echo session_name() ?>=<?php echo uniqid() ?>"></p>
        <br>
        <label for="code">Код подтверждения</label>
        <input name="code" id="code" />
        <br>
        <br>
        <button type="submit">Отправить</button>
    </form>
    </div>

</body>

</html>