#include "FiringPlot.hpp"

#include <QPainter>
#include <QTimer>
#include <QResizeEvent>
#include <QTextStream>

//! \todo remove
#include <iostream>
#include <assert.h>


FiringPlot::FiringPlot(int fps, QWidget* parent) :
	QWidget(parent),
	m_fps(fps),
	m_updates(0),
	m_fbuffer(1), //! \todo set this based on size
	m_neuronCount(0),
	m_yscale(1.0),
	m_temporalResolution(10),
	m_cycle(-1),
	m_fillBuffer(0),
	m_fillCycle(0),
	m_axisPadding(0),
	m_axisTickSize(4),
	m_axisFontSize(12)
{
	setMinimumSize(320, 240);
	setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	setAttribute(Qt::WA_OpaquePaintEvent);
	//setAttribute(Qt::WA_PaintOnScreen); // faster, but more flicker
	
	incrementTime(1);

#if 0
	//! \todo remove timer: drive display by data instead
	QTimer* timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(addSpikes()));
	timer->setInterval(1000/m_fps);
	timer->start();
#endif
#if 1
	QTimer::singleShot(10, this, SLOT(readStdin()));
	m_runtime.start();
#endif
}



FiringPlot::~FiringPlot()
{
	std::cout << "Frame rate (target=" << m_fps << "): " 
		<< 1000*m_updates/m_runtime.elapsed() << std::endl;
}



void
FiringPlot::readStdin()
{
	//! \todo keep track of current time
	// whenever the current data is greater than m_cycle +
	// m_temporalResolution, draw to screen.
	QTextStream stream(stdin);

	std::cerr << "Starting stdin processing\n";
	
	// The first value should specify number of neurons to plot
	stream >> m_neuronCount;
	std::cerr << "Network size: " << m_neuronCount << std::endl;
	resizeBuffers();
	
	do {
		int neuron, cycle;
		stream >> cycle;
		stream >> neuron;
		if(cycle < m_cycle) {
			/* This is probably not the most sensible way to exit this loop.
			 * Using stream.atEnd() does not work for stdin.  Using readLine
			 * and then line.isNull could work, but we'd have to parse that
			 * separately. It seems that stdin gives us all zeros when there's
			 * no more data, so current implementation should work in most
			 * cases. */
			break;
		}
		if(cycle > m_cycle) {
			//! \todo use setTime instead?
			incrementTime(cycle - m_cycle);
			// wait for repaint!
			//msleep(1);
		}
		if(neuron != -1) {
			addSpike(cycle, neuron);
		}
	} while (1);

	std::cerr << "Stopped processing\n";
}



void
FiringPlot::incrementTime(int cycles)
{
	assert(cycles > 0);

	m_cycle += cycles;

	while(m_cycle >= m_fillCycle + m_temporalResolution) {

		//! \todo keep track of last repaint time
		//! \todo use update, but yield control
		//update();
		repaint();

		m_fillCycle += m_temporalResolution;	
		m_fillBuffer = (m_fillBuffer + 1) % m_fbuffer.size();

		QImage& buffer = m_fbuffer[m_fillBuffer];
		//! \todo the fill seems to have no effect. Find out why and then remove.
		buffer.fill(Qt::black);

		if(m_fillPainter.isActive()){
			m_fillPainter.end();
		}
		QPen pen(Qt::red);
		pen.setWidth(2);
		m_fillPainter.begin(&buffer);
		m_fillPainter.setPen(pen);
		m_fillPainter.scale(1.0, m_yscale);
		m_fillPainter.setRenderHint(QPainter::Antialiasing);
	}
}



void
FiringPlot::addSpike(int cycle, int neuron)
{
	assert(cycle >= m_fillCycle);
	assert(cycle < m_fillCycle + m_temporalResolution);
	assert(neuron < m_neuronCount);

	if(cycle >= m_fillCycle && 
			cycle < m_fillCycle + m_temporalResolution &&
			neuron < m_neuronCount) {
		m_fillPainter.drawPoint(cycle-m_fillCycle, neuron);
	} else {
		assert(false);
	}
}



void
FiringPlot::addSpikes()
{
	for(int i=0; i<8; ++i){
		incrementTime(m_temporalResolution);
		for(int j=0; j<100; ++j) {
			int cycle = qrand() % m_temporalResolution;
			int neuron = qrand() % m_neuronCount; 
			addSpike(m_fillCycle + cycle, neuron);
		}
	}
}



void
FiringPlot::resizeBuffers()
{
	int bufferWidth = m_temporalResolution;
	//int bufferHeight = widgetHeight; // - m_axisPadding;
	int bufferHeight = height(); // - m_axisPadding;
	//int bufferCount = widgetWidth / bufferWidth;
	int bufferCount = width() / bufferWidth;

	/*! \note if there are more neurons than there are pixels in the
	 * y-dimension, we might end up with spikes having less than a pixel in
	 * which case they are not visible. */
	m_yscale = std::max(1.0, qreal(bufferHeight) / qreal(m_neuronCount));

	//! \todo resize existing data in the pipeline?
	m_fbuffer.clear();

	//! \todo no need to create buffers here, can create as needed!
	for(int b=0; b < bufferCount; ++b) {
		QImage buffer(bufferWidth, bufferHeight, QImage::Format_RGB32);
		buffer.fill(Qt::black);
		m_fbuffer.push_back(buffer);
	}
}


//! \todo set width properly
void 
FiringPlot::resizeEvent(QResizeEvent* /*event*/)
{
	/* The temporal resolution is fixed at 1px/ms. */
	//! \todo may need to deal with different temporal resolution in the
	//simulation

	//! \todo separate adding data and displaying data. Only display data at the speed at which it is produced.
	//int widgetHeight = event->size().height();
	//int widgetWidth = event->size().width();
	resizeBuffers();
}



/*! \param yval
 * 		unscaled y-value
 * 	\param label
 * 		if set, also show the value and make this a major tick. If unset make
 * 		this a minor tick.
 */
void
FiringPlot::paintYAxisTick(QPainter& painter, int valY, bool label)
{
	int tickY = valY * m_yscale;
	int tickW = label ? m_axisTickSize : m_axisTickSize/2;
	painter.drawLine(m_axisPadding, tickY, m_axisPadding + tickW, tickY);
	if(label) {
		int textY = tickY - m_axisFontSize/2;
		painter.drawText(m_axisPadding + m_axisTickSize + 1, textY, 
				m_axisFontSize*5, m_axisFontSize,
				Qt::AlignLeft | Qt::AlignVCenter,
				QString::number(valY));
	}
}



void
FiringPlot::paintYAxis(QPainter& painter)
{
	for( int i=1; i < 8; ++i ){
		paintYAxisTick(painter, i*m_neuronCount/8, i % 2 == 0);
	}
}



void
FiringPlot::paintXAxisTick(QPainter& painter, int valX, int labelX)
{
	// no scaling along the x-axis
	painter.drawLine(valX, height()-1, valX, height()-1-m_axisTickSize);
	//! \todo perhaps use all the space we can
	int textW = m_axisFontSize*5;
	painter.drawText(valX - textW/2, height()-2-m_axisTickSize-m_axisFontSize, 
			textW, m_axisFontSize,
			Qt::AlignHCenter,
			QString("%1ms").arg(labelX) );
			//QString::number(labelX));
}



/*! set axis ticks for every 50ms */
void
FiringPlot::paintXAxis(QPainter& painter)
{
	//! \todo assert that the interval is a multiple of m_temporalResolution
	int startCycle = m_fillCycle - m_temporalResolution * m_fbuffer.size();
	for(int i=0; i < m_fbuffer.size(); ++i){
		//int currentBuffer = (firstBuffer() + i ) % m_fbuffer.size();
		int timeOffset = i * m_temporalResolution;
		int time = startCycle + timeOffset;
		if(time % 100 == 0) {
			paintXAxisTick(painter, timeOffset, time);
		}
		//painter.drawImage(i*w, 0, m_fbuffer[currentBuffer]);
	}
}



int
FiringPlot::firstBuffer() const
{
	return (m_fillBuffer + 1 ) % m_fbuffer.size();
}



void
FiringPlot::paintAxes(QPainter& painter)
{
	painter.setPen( QPen(Qt::white) );

	QFont font; 
	font.setFamily("Helvetica");
	font.setWeight(QFont::Light);
	font.setPixelSize(m_axisFontSize);
	painter.setFont(font);

	paintYAxis(painter);
	paintXAxis(painter);
}



void
FiringPlot::paintData(QPainter& painter)
{
	int w = m_fbuffer.first().width();
	for(int i=0; i < m_fbuffer.size(); ++i){
		int currentBuffer = (firstBuffer() + i ) % m_fbuffer.size();
		painter.drawImage(i*w, 0, m_fbuffer[currentBuffer]);
	}
}



void 
FiringPlot::paintEvent(QPaintEvent* /* event */)
{
	m_updates += 1;
	QPainter painter(this);
	paintData(painter);
	paintAxes(painter);
	painter.end();
}
