package ui_components;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionListener;

public class ButtonsPanel extends JPanel {
    private final JButton addButton = new JButton("Insert");
    private final JButton removeButton = new JButton("Remove");
    private final JButton stopButton = new JButton("Stop/Resume");
    private final JButton stopAllButton = new JButton("Stop all");
    private final JButton resumeAllButton = new JButton("Resume all");

    public ButtonsPanel() {
        setLayout(new GridLayout(1, 5));
        add(addButton);
        add(removeButton);
        add(stopButton);
        add(stopAllButton);
        add(resumeAllButton);
    }

    public void onAddButtonClicked(ActionListener listener) {
        addButton.addActionListener(listener);
    }

    public void onRemoveButtonClicked(ActionListener listener) {
        removeButton.addActionListener(listener);
    }

    public void onStopButtonClicked(ActionListener listener) {
        stopButton.addActionListener(listener);
    }

    public void onStopAllButtonClicked(ActionListener listener) {
        stopAllButton.addActionListener(listener);
    }

    public void onResumeAllButtonClicked(ActionListener listener) {
        resumeAllButton.addActionListener(listener);
    }
}