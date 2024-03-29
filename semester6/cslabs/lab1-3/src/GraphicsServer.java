import javax.swing.*;

public class GraphicsServer {
    public static void main(String[] args) {
        final var frame = new JFrame("Server");
        final var editor = new GraphicsController(true);
        frame.setContentPane(editor);
        frame.setSize(700, 900);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}


class GraphicsClient {
    public static void main(String[] args) {
        final var frame = new JFrame("Client");
        final var editor = new GraphicsController(false);
        frame.setContentPane(editor);
        frame.setSize(700, 900);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}