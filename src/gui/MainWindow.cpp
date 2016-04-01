#include "MainWindow.hpp"
#include "FiringPlot.hpp"

#include <QAction>
#include <QLabel>
#include <QImage>


MainWindow::MainWindow()
{
	createActions();
	m_plot = new FiringPlot(25, this);
	setCentralWidget(m_plot);
	showMaximized();
}



void
MainWindow::createActions()
{
	m_exitAct = new QAction(tr("E&xit"),this);
	m_exitAct->setShortcut(tr("Ctrl+Q"));
	connect(m_exitAct, SIGNAL(triggered()), this, SLOT(close()));
	this->addAction(m_exitAct);
}
