package objects;

import org.json.JSONObject;
import utils.Vector;

import java.awt.*;
import java.util.Random;

public class Smiley extends GraphicalObject {
    private final Random random = new Random();
    private int vx = random.nextInt(1, 7);
    private int vy = random.nextInt(1, 7);


    public Smiley(int x, int y, int width, int height, Color color) {
        super(x, y, width, height, color);
    }

    @Override
    public void draw(Graphics g) {
        super.draw(g);
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

    @Override
    public void readFromJson(String json) {
        var jsonObject = new JSONObject(json);
        x = jsonObject.getInt("x");
        y = jsonObject.getInt("y");
        width = jsonObject.getInt("width");
        height = jsonObject.getInt("height");
        color = new Color(jsonObject.getInt("r"), jsonObject.getInt("g"), jsonObject.getInt("b"));
        vx = jsonObject.getInt("vx");
        vy = jsonObject.getInt("vy");
    }

    @Override
    public String writeToJson() {
        var jsonObject = new JSONObject();
        jsonObject.put("x", x);
        jsonObject.put("y", y);
        jsonObject.put("width", width);
        jsonObject.put("height", height);
        jsonObject.put("r", color.getRed());
        jsonObject.put("g", color.getGreen());
        jsonObject.put("b", color.getBlue());
        jsonObject.put("vx", vx);
        jsonObject.put("vy", vy);
        return jsonObject.toString();
    }

    @Override
    public String toString() {
        return "Smiley{" +
                "x=" + x +
                ", y=" + y +
                ", width=" + width +
                ", height=" + height +
                ", color=" + color +
                '}';
    }
}


