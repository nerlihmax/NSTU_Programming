import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        final var frame = new JFrame("Server");
        final var editor = new GraphicsController(true);
        frame.setContentPane(editor);
        frame.setSize(800, 800);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}


class Main2 {
    public static void main(String[] args) {
        final var frame = new JFrame("Client");
        final var editor = new GraphicsController(false);
        frame.setContentPane(editor);
        frame.setSize(800, 800);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}