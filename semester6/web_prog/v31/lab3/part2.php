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
    <title>Лабораторная №3 часть 2xxs</title>
</head>

<body>
    <form action="journal.php" method="post" target="_self">
        <h3>Приветствие: </h3>
        <div id="ckeditor">
            <?php
            $oFCKeditor = new FCKeditor('text_input');
            $oFCKeditor->BasePath = './fckeditor/';
            $oFCKeditor->Value = '<b>Привет, мир!</b>';
            $oFCKeditor->Create();
            ?>
        </div>
        <br>

        <h3>Выберите стиль приветствия:</h3>
        <div>
            <div>
                <input type="radio" name="style" value="Официальное" id="21" checked="checked">
                <label for="21">Официальное</label>
            </div>

            <div>
                <input type="radio" name="style" id="22" value="Неофициальное">
                <label for="22">Неофициальное</label>
            </div>

            <div>
                <input type="radio" name="style" value="Дружеское" id="23">
                <label for="23">Дружеское</label>
            </div>
        </div>

        <h3>Выберите размер шрифта: </h3>
        <select name="size">
            <option value="10 pt" selected="selected">Маленький (10pt)</option>
            <option value="18 pt">Средний (18pt)</option>
            <option value="32 pt">Большой (32pt)</option>
        </select>

        <br>
        <h3>Капча</h3>
        <img src="/lab3/kcaptcha/index.php?<?php echo session_name() ?>=<?php echo uniqid() ?>"></p>
        <br>
        <label for="code">Введите капчу: </label>
        <input name="code" id="code" />
        <br>
        <br>
        <button type="submit">Отправить</button>
    </form>
    </div>

</body>

</html>