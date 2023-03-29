import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        final var frame = new JFrame("REST API CLIENT");
        final var editor = new GraphicsController(true);
        frame.setContentPane(editor);
        frame.setSize(900, 600);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}