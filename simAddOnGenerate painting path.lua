function sysCall_info()
    return {autoStart=false}
end

function sysCall_init()
	corout=coroutine.create(coroutineMain)
    sim.addLog(sim.verbosity_scriptinfos,"Select an object for which to generate a painting path and press \"Select faces\". A window will open for you to select painting areas.")
	nozzle_handle = "/PaintNozzle"
	jetAngle=-1
    jetRange=-1
	obj=-1
end

function resumeCoroutine()
	if coroutine.status(corout)~='dead' then
        local ok,errorMsg=coroutine.resume(corout)
        if errorMsg then
            error(debug.traceback(corout,errorMsg),2)
        end
    end
end

function sysCall_addOnScriptSuspend()
    return {cmd='cleanup'}
end

function showDlg()
    if not ui then
        local pos='position="-50,50" placement="relative"'
        if uiPos then
            pos='position="'..uiPos[1]..','..uiPos[2]..'" placement="absolute"'
        end
        local xml ='<ui title="Painting path generation" activate="false" closeable="true" on-close="close_callback" layout="vbox" '..pos..[[>
				<label text="Enter paint nozzle name (has to be unique):"/>
				<edit id="1" on-editing-finished="nozzle_handle_callback" value="PaintNozzle"/>
				<label text="Jet angle"/>
                <edit id="2"/>
                <label text="Jet range"/>
                <edit id="3"/>
				<button text="Select faces to paint..." on-click="select_callback"/>
        </ui>]]
        ui=simUI.create(xml)
		jetAngle=get_nozzle_value("jetAngle")
		jetRange=get_nozzle_value("jetRange")
		simUI.setEditValue(ui,2,tostring(jetAngle))
		simUI.setEditValue(ui,3,tostring(jetRange))
    end
end

function hideDlg()
    if ui then
        uiPos={}
        uiPos[1],uiPos[2]=simUI.getPosition(ui)
        simUI.destroy(ui)
        ui=nil
    end
    jetAngle=-1
    jetRange=-1
    obj=-1
end

function sysCall_nonSimulation()
	resumeCoroutine()
    if leaveNow then
        return {cmd='cleanup'}
    end
    showOrHideDlg()
end

function sysCall_actuation()
    resumeCoroutine()
end

function coroutineMain()
	--sim.launchExecutable('pathGenerationZmqServer',"23000 0.2 0.15",0)
end

function sysCall_beforeSimulation()
    hideDlg()
end

function sysCall_cleanup()
    hideDlg()
end

function sysCall_beforeInstanceSwitch()
    hideDlg()
end

function showOrHideDlg()
    local s=sim.getObjectSelection()
    local show=(s and #s==1 and sim.getObjectType(s[1])==sim.object_shape_type)
    if show  then
        if obj~=s[1] then
            hideDlg()
            obj=s[1]
        end
        showDlg()
    else
        obj=-1
        hideDlg()
    end
end

function get_nozzle_value(v)
	result, execRes = sim.executeScriptString("return "..v.."@"..nozzle_handle, 6, 0)
	return execRes
end

function nozzle_handle_callback(ui, id, v)
	nozzle_handle = "/"..v
end

function select_callback(ui, id, v)
	simAssimp.exportShapes({obj}, "exported.obj", "objnomtl")
	
	local height=jetRange
	local angle1=jetAngle / 2
	local angle2= math.pi/2 - angle1
	local width=math.floor(2*(height*math.sin(angle1)*math.sin(angle2)) * 1000) / 1000
	
	local portNb=sim.getInt32Param(sim.intparam_server_port_next)
    local portStart=sim.getInt32Param(sim.intparam_server_port_start)
    local portRange=sim.getInt32Param(sim.intparam_server_port_range)
    local newPortNb=portNb+1
    if (newPortNb>=portStart+portRange) then
        newPortNb=portStart
    end
    sim.setInt32Param(sim.intparam_server_port_next,newPortNb)
	local handle = simSubprocess.execAsync("pathGenerationZmqServer", {"23000", 
																	   tostring(width),
																	   tostring(height),
																	   "--sp", "0.05", "0.0054", "1.215",
																	   "--so", "0.0", "0.0", "-1.0", "0.0"})
	print(handle)
	
	--leaveNow=true

end

function close_callback()
    leaveNow=true
end

