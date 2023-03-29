package ui_components;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionListener;

public class ButtonsPanel extends JPanel {
    private final JButton addButton = new JButton("Insert"),
            removeButton = new JButton("Remove"),
            stopButton = new JButton("Stop/Resume"),
            stopAllButton = new JButton("Stop all"),
            resumeAllButton = new JButton("Resume all"),
            clearAllButton = new JButton("Clear");

    public ButtonsPanel() {
        setLayout(new GridLayout(1, 6));
        add(addButton);
        add(removeButton);
        add(stopButton);
        add(stopAllButton);
        add(resumeAllButton);
        add(clearAllButton);
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

    public void onClearButtonClicked(ActionListener listener) {
        clearAllButton.addActionListener(listener);
    }
}