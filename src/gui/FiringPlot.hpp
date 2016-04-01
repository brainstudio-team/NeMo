#ifndef FIRING_PLOT_HPP 
#define FIRING_PLOT_HPP

#include <QWidget>
#include <QVector>
#include <QImage>
#include <QTime>
#include <QPainter>


/*! \brief Widget for plotting firing data
 *
 * Firing data can arrive at different rates. The data is plotted at /at most/
 * the framerate specified in the ctor. 
 *
 * \author Andreas Fidjeland
 */
class FiringPlot : public QWidget
{
	Q_OBJECT

	public :

		FiringPlot(int fps, QWidget* parent=0);

		~FiringPlot();

	public slots :

		/*! Increment time, any new spikes should only have time-stamps /after/
		 * the current time after the update */
		void incrementTime(int cycles=1);

		/*! Spikes are ignored if they are out of bounds, either spatially or
		 * temporally */
		void addSpike(int time, int neuron);

		//! \todo same, but for a vector of neurons

		//! \todo remove
		//! Create 5ms worth of random spikes, for debugging only
		void addSpikes();

	protected:

		void paintEvent(QPaintEvent* event);

		void resizeEvent(QResizeEvent* event);

	private slots :

		/*! read data from stdin and pass on to display */
		void readStdin();

	private:

		int m_fps;

		QVector< QPair<int, int> > m_spikes;

		QTime m_runtime;
		int m_updates;

		// buffer each slice independently, only update the last one
		//! \todo deal with resizing here
		QVector<QImage> m_fbuffer;

		//! \todo allow changing neuron count?
		int m_neuronCount;
		qreal m_yscale;

		/*! \todo might adjust this depending on the temporal resolution of the
		 * simulation. */
		int m_temporalResolution; // for display, not for running

		//! \todo avoid overflow!
		/*! Last cycle for which we have complete data */
		int m_cycle;

		/*! Buffer to which new data should be added, corresponding to the
		 * interval m_fillCycle to m_fillCycle + m_temporalResolution */
		int m_fillBuffer;
		int m_fillCycle;
		QPainter m_fillPainter;

		void paintData(QPainter&);

		/*! \return the index of the first buffer in the queue */
		int firstBuffer() const;

		void resizeBuffers();

		/*
		 * Axis
		 */

		int m_axisPadding;
		int m_axisTickSize;
		int m_axisFontSize; // in pixels

		void paintYAxis(QPainter&);
		void paintYAxisTick(QPainter&, int yval, bool label);

		void paintXAxis(QPainter&);
		void paintXAxisTick(QPainter&, int xval, int xlable);

		void paintAxes(QPainter&);

};

#endif
