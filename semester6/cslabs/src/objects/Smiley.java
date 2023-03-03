package objects;

import utils.Vector;

import java.awt.*;

public class Smiley extends GraphicalObject {
    private int vx = 3;
    private int vy = 3;

    public Smiley(int x, int y, int width, int height, Color color) {
        super(x, y, width, height, color);
    }

    @Override
    public void draw(Graphics g) {
        super.draw(g);
        Graphics2D g2d = (Graphics2D) g;

        g.setColor(color);
        g.fillOval(x - width / 2, y - height / 2, width, height);
        g.setColor(Color.BLACK);
        g.drawOval(x - width / 2, y - height / 2, width, height);
        g.drawOval(x - width / 3, y - height / 3, width / 6, height / 6);
        g.drawOval(x + width / 6, y - height / 3, width / 6, height / 6);
        g.drawArc(x - width / 4, y - height / 4, width / 2, height / 2, 190, 160);
    }

    @Override
    public void move(Vector canvas) {
        if (x + width / 2 >= canvas.x() || x - width / 2 <= 0) {
            vx *= -1;
        }
        if (y + height / 2 >= canvas.y() || y - height / 2 <= 0) {
            vy *= -1;
        }
        x += vx;
        y += vy;
    }
}


