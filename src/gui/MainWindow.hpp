#ifndef MAIN_WINDOW_HPP
#define MAIN_WINDOW_HPP

#include <QMainWindow>

class QAction;
class FiringPlot;

class MainWindow : public QMainWindow
{
	Q_OBJECT

	public :

		MainWindow();

	private :

		void createActions();

		QAction* m_exitAct;

		FiringPlot* m_plot;
};


#endif
