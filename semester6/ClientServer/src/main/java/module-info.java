module ru.kheynov.clientserver {
    requires javafx.controls;
    requires javafx.fxml;


    opens ru.kheynov.clientserver to javafx.fxml;
    exports ru.kheynov.clientserver;
}