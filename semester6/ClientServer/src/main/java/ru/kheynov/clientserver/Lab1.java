package ru.kheynov.clientserver;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class Lab1 extends Application {
    private int count = 1;

    private static final int WIDTH = 600;
    private static final int HEIGHT = 400;

    @Override
    public void start(Stage stage) {
        initUI(stage);
    }

    private void initUI(Stage stage) {
        var root = new Pane();
        var canvas = new Canvas(WIDTH, HEIGHT);
        var gc = canvas.getGraphicsContext2D();
        drawLines(gc);

        root.getChildren().add(canvas);

        var scene = new Scene(root, WIDTH, HEIGHT, Color.WHITESMOKE);

        Thread thread = new Thread(() -> {
            Runnable updater = () -> {
                count++;
                drawLines(gc);
            };
            while (true) {
                try {
                    Thread.sleep(20);
                } catch (InterruptedException ignored) {
                }
                Platform.runLater(updater);
            }
        });
        thread.setDaemon(true);
        thread.start();

        stage.setTitle("Lines");
        stage.setScene(scene);
        stage.show();

    }

    private void drawLines(GraphicsContext gc) {
        gc.clearRect(0, 0, WIDTH, HEIGHT);

        gc.beginPath();
        gc.moveTo(30.5 + count, 30.5 + count);
        gc.lineTo(150.5, 30.5);
        gc.lineTo(150.5, 150.5);
        gc.lineTo(30.5, 30.5);
        gc.stroke();
    }

    public static void main(String[] args) {
        launch(args);
    }
}