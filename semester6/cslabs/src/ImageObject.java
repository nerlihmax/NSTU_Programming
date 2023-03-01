import javax.imageio.ImageIO;
import java.awt.*;
import java.io.*;
import java.net.URL;

public class ImageObject extends GraphicalObject {
    private Image image;

    private int vx = 3;
    private int vy = 3;

    public ImageObject(int x, int y, int width, int height, Color color, String imageUrl) throws IOException {
        super(x, y, width, height, color);
        final var url = new URL(imageUrl);
        image = ImageIO.read(url);
    }

    @Override
    public void draw(Graphics g, int canvasWidth, int canvasHeight) {
        g.drawImage(image, x - width / 2, y - height / 2, width, height, null);
    }

    @Override
    public void read(InputStream input) throws IOException {
        super.read(input);
        final var dis = new DataInputStream(input);
        final var imageUrl = dis.readUTF();
        final var url = new URL(imageUrl);
        image = ImageIO.read(url);
    }

    @Override
    public void write(OutputStream output) throws IOException {
        super.write(output);
        final var dos = new DataOutputStream(output);
        dos.writeUTF(image.toString());
    }

    @Override
    public void move(Vector movement) {
        if (x + width / 2 >= movement.x() || x - width / 2 <= 0) {
            vx *= -1;
        }
        if (y + height / 2 >= movement.y() || y - height / 2 <= 0) {
            vy *= -1;
        }
        x += vx;
        y += vy;
    }
}
