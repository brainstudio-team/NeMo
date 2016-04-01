#include <algorithm>
#include <cassert>

#include "exception.hpp"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {


template<typename T>
NVector<T>::NVector(
		size_t planes,
		size_t partitionCount,
		size_t maxPartitionSize,
		bool allocHostData,
		bool pinHostData) :
	m_planes(planes),
	m_partitionCount(partitionCount),
	m_pitch(0)
{
	size_t height = planes * partitionCount;
	if(height == 0 || maxPartitionSize == 0) {
		/* Empty array, leave this as null pointer and 0 size */
		return;
	}

	size_t bytePitch = 0;
	void* d_ptr = NULL;
	d_mallocPitch(&d_ptr, &bytePitch, maxPartitionSize * sizeof(T), height, "NVector");
	m_deviceData = boost::shared_array<T>(static_cast<T*>(d_ptr), d_free);

	m_pitch = bytePitch / sizeof(T);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	d_memset2D(d_ptr, bytePitch, 0x0, height);

	//! \todo may need a default value here
	if(allocHostData) {
		if(pinHostData) {
			void* h_ptr = NULL;
			mallocPinned(&h_ptr, height * m_pitch * sizeof(T));
			m_hostData = boost::shared_array<T>(static_cast<T*>(h_ptr), freePinned);
		} else {
			m_hostData = boost::shared_array<T>(new T[height * m_pitch]);
		}
		std::fill(m_hostData.get(), m_hostData.get() + height * m_pitch, 0x0);
	}
}



template<typename T>
T*
NVector<T>::deviceData() const
{
	return m_deviceData.get();
}


template<typename T>
size_t
NVector<T>::size() const
{
	return m_partitionCount * m_pitch;
}


template<typename T>
size_t
NVector<T>::bytes() const
{
	return m_planes * size() * sizeof(T);
}


template<typename T>
size_t
NVector<T>::d_allocated() const
{
	return bytes();
}



template<typename T>
size_t
NVector<T>::wordPitch() const
{
	return m_pitch;
}


template<typename T>
size_t
NVector<T>::bytePitch() const
{
	return m_pitch * sizeof(T);
}


template<typename T>
const T*
NVector<T>::copyFromDevice()
{
	memcpyFromDevice(m_hostData.get(), m_deviceData.get(), m_planes * size());
	return m_hostData.get();
}


template<typename T>
void
NVector<T>::moveToDevice()
{
	copyToDevice();
	m_hostData.reset();
}


template<typename T>
void
NVector<T>::copyToDevice()
{
	if(!empty()) {
		memcpyToDevice(m_deviceData.get(), m_hostData.get(), m_planes * size());
	}
}



template<typename T>
void
NVector<T>::copyToDeviceAsync(cudaStream_t stream)
{
	memcpyToDeviceAsync(m_deviceData.get(), m_hostData.get(), m_planes * size(), stream);
}


template<typename T>
size_t
NVector<T>::offset(size_t subvector, size_t partitionIdx, size_t neuronIdx) const
{
	//! \todo throw exception if incorrect size is used
	assert(subvector < m_planes);
	assert(partitionIdx < m_partitionCount);
	assert(neuronIdx < m_pitch);
	return (subvector * m_partitionCount + partitionIdx) * m_pitch + neuronIdx;
}



template<typename T>
void
NVector<T>::setNeuron(size_t partitionIdx, size_t neuronIdx, const T& val, size_t subvector)
{
    m_hostData[offset(subvector, partitionIdx, neuronIdx)] = val;
}



template<typename T>
void
NVector<T>::set(const std::vector<T>& vec)
{
	assert(vec.size() == m_planes * size());
	std::copy(vec.begin(), vec.end(), m_hostData.get());
}



template<typename T>
T
NVector<T>::getNeuron(size_t partitionIdx, size_t neuronIdx, size_t subvector) const
{
    return m_hostData[offset(subvector, partitionIdx, neuronIdx)];
}



template<typename T>
void
NVector<T>::fill(const T& val, size_t subvector)
{
	std::fill(m_hostData.get() + subvector * size(), m_hostData.get() + (subvector+1) * size(), val);
}



template<typename T>
void
NVector<T>::replicateInitialPlanes(size_t n)
{
	T* base = m_hostData.get();
	for(size_t tgt=n; tgt < m_planes; tgt += n) {
		std::copy(base, base+n*size(), base+tgt*size());
	}
}


}	} // end namespace
