import network.NetworkEventListener;
import network.NetworkRepository;
import network.TCPNetworkRepository;
import objects.GraphicalObject;
import objects.Smiley;
import objects.Star;
import ui_components.ButtonsPanel;
import utils.EditorModes;
import utils.Vector;
import utils.network_events.NetworkEvent;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GraphicsController extends JPanel implements NetworkEventListener {
    private final ArrayList<GraphicalObject> objects = new ArrayList<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final ScheduledExecutorService scheduledExecutor = Executors.newSingleThreadScheduledExecutor();
    private final Random random = new Random();
    private final NetworkRepository networkRepository;

    private EditorModes mode = EditorModes.ADD;

    public GraphicsController(boolean isServer) {
        registerModesPanel();
        start();
        networkRepository = new TCPNetworkRepository(isServer, this);
        var thread = new Thread((Runnable) networkRepository);
//        thread.setDaemon(true);
        thread.start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (GraphicalObject object : objects) {
            object.draw(g);
        }

    }

    private void registerModesPanel() {
        ButtonsPanel buttonsPanel = new ButtonsPanel();
        add(buttonsPanel, BorderLayout.SOUTH);

        buttonsPanel.onAddButtonClicked(e -> {
            mode = EditorModes.ADD;
            objects.forEach(GraphicalObject::hideOutline);
        });

        buttonsPanel.onRemoveButtonClicked(e -> {
            mode = EditorModes.REMOVE;
            objects.forEach(GraphicalObject::showOutline);
        });

        buttonsPanel.onStopButtonClicked(e -> {
            mode = EditorModes.STOP_RESUME;
            objects.forEach(GraphicalObject::showOutline);
        });

        buttonsPanel.onStopAllButtonClicked(e -> objects.forEach(GraphicalObject::stop));

        buttonsPanel.onResumeAllButtonClicked(e -> objects.forEach(GraphicalObject::resume));
    }

    private void start() {
        scheduledExecutor.scheduleAtFixedRate(() -> {
            executor.submit(() -> objects.forEach(obj -> {
                if (obj.isMoving()) {
                    obj.move(new Vector(getWidth(), getHeight()));
                }
            }));

            SwingUtilities.invokeLater(this::repaint);
        }, 0, 16, TimeUnit.MILLISECONDS);

        setBackground(Color.WHITE);
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();

                GraphicalObject object = null;

                if (mode == EditorModes.REMOVE) {
                    objects.removeIf(obj -> obj.contains(x, y));
                    return;
                }

                if (mode == EditorModes.STOP_RESUME) {
                    objects.stream().filter(obj -> obj.contains(x, y)).findFirst().ifPresent(obj -> {
                        if (obj.isMoving()) {
                            obj.stop();
                        } else {
                            obj.resume();
                        }
                    });
                    return;
                }
                if (e.getButton() == MouseEvent.BUTTON1) {
                    object = new Smiley(x, y, 50, 50, Color.CYAN);
                    networkRepository.sendObject(object);
                } else if (e.getButton() == MouseEvent.BUTTON3) {
                    var size = random.nextInt(50, 150);
                    var vertices = random.nextInt(5, 9);
                    object = new Star(x, y, size, size, Color.RED, vertices);
                }

                if (object != null) {
                    objects.add(object);
                }
            }
        });
    }

    @Override
    public void onEvent(NetworkEvent event) {

    }
}