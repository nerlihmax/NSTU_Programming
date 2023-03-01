import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.*;
import java.util.ArrayList;

public class GraphicsController extends JPanel {
    private ArrayList<GraphicalObject> objects = new ArrayList<>();

    public GraphicsController() {
        setBackground(Color.WHITE);
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();
                GraphicalObject object = null;
                if (e.getButton() == MouseEvent.BUTTON1) {
                    object = new Smiley(x, y, 50, 50, Color.GREEN);
                } else if (e.getButton() == MouseEvent.BUTTON3) {
                    object = new Smiley(x, y, 50, 50, Color.RED);
                }
                if (object != null) {
                    objects.add(object);
                }
            }
        });

        // Screen updates ~60 FPS
        SwingUtilities.invokeLater(() -> {
            var timer = new Timer(10, e -> {
                // Move the objects
                for (GraphicalObject obj : objects) {
                    obj.move(new Vector(getWidth(), getHeight()));
                }
                // Repaint the canvas
                repaint();
            });
            timer.start();
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (GraphicalObject object : objects) {
            object.draw(g, getWidth(), getHeight());
        }
    }

    public void saveToFile(String fileName) throws IOException {
        OutputStream output = new BufferedOutputStream(new FileOutputStream(fileName));
        ObjectOutputStream oos = new ObjectOutputStream(output);
        oos.writeObject(objects);
    }

    public void loadFromFile(String fileName) throws IOException {
        try (InputStream input = new BufferedInputStream(new FileInputStream(fileName))) {
            ObjectInputStream ois = new ObjectInputStream(input);
            try {
                objects = (ArrayList<GraphicalObject>) ois.readObject();
            } catch (Exception e) {
                System.out.println(e.getLocalizedMessage());
            }
        }
        repaint();
    }
}