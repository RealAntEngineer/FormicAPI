package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;
import org.jocl.cl_event;

public record OpenCLEvent(cl_event event) implements ResourceEvent {
}
