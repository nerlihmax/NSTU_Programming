import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        final var frame = new JFrame("Client Server Labs 2-3");
        final var editor = new GraphicsController();
        frame.setContentPane(editor);
        frame.setSize(800, 800);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}