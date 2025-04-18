﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on März 25, 2025, at 14:23
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'trigger_test'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Nemecz\\Documents\\Guess_fMRI-main\\trigger_test_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('move_on') is None:
        # initialise move_on
        move_on = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='move_on',
        )
    if deviceManager.getDevice('button_press') is None:
        # initialise button_press
        button_press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='button_press',
        )
    if deviceManager.getDevice('select_pin') is None:
        # initialise select_pin
        select_pin = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='select_pin',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "button_test" ---
    move_on = keyboard.Keyboard(deviceName='move_on')
    button_press = keyboard.Keyboard(deviceName='button_press')
    info_press = visual.TextStim(win=win, name='info_press',
        text='Pressed button:',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    pressed_text = visual.TextStim(win=win, name='pressed_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from catch_button
    button = 'None'
    
    # --- Initialize components for Routine "read_port" ---
    # Run 'Begin Experiment' code from catch_signal
    from psychopy import parallel
    port = parallel.ParallelPort(address = 0x2FE8) 
    counter10 = 1
    counter12 = 1
    read_port_text = visual.TextStim(win=win, name='read_port_text',
        text='Reading the Port\nPress the number of the pin you want to listen to.',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    select_pin = keyboard.Keyboard(deviceName='select_pin')
    display_trigger2 = visual.TextStim(win=win, name='display_trigger2',
        text='',
        font='Arial',
        pos=(-0.5, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    display_trigger3 = visual.TextStim(win=win, name='display_trigger3',
        text='',
        font='Arial',
        pos=(-0.4, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    display_trigger4 = visual.TextStim(win=win, name='display_trigger4',
        text='',
        font='Arial',
        pos=(-0.3, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    display_trigger5 = visual.TextStim(win=win, name='display_trigger5',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    display_trigger6 = visual.TextStim(win=win, name='display_trigger6',
        text='',
        font='Arial',
        pos=(-0.1, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    display_trigger7 = visual.TextStim(win=win, name='display_trigger7',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    display_trigger8 = visual.TextStim(win=win, name='display_trigger8',
        text='',
        font='Arial',
        pos=(0.1, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    display_trigger9 = visual.TextStim(win=win, name='display_trigger9',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    display_trigger10 = visual.TextStim(win=win, name='display_trigger10',
        text='',
        font='Arial',
        pos=(0.3, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    display_trigger11 = visual.TextStim(win=win, name='display_trigger11',
        text='',
        font='Arial',
        pos=(0.4, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    display_trigger12 = visual.TextStim(win=win, name='display_trigger12',
        text='',
        font='Arial',
        pos=(0.5, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    
    # --- Initialize components for Routine "pin10" ---
    counter10_text2 = visual.TextStim(win=win, name='counter10_text2',
        text='',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "pin12" ---
    counter12_text2 = visual.TextStim(win=win, name='counter12_text2',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "button_test" ---
    # create an object to store info about Routine button_test
    button_test = data.Routine(
        name='button_test',
        components=[move_on, button_press, info_press, pressed_text],
    )
    button_test.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for move_on
    move_on.keys = []
    move_on.rt = []
    _move_on_allKeys = []
    # create starting attributes for button_press
    button_press.keys = []
    button_press.rt = []
    _button_press_allKeys = []
    # store start times for button_test
    button_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    button_test.tStart = globalClock.getTime(format='float')
    button_test.status = STARTED
    thisExp.addData('button_test.started', button_test.tStart)
    button_test.maxDuration = None
    # keep track of which components have finished
    button_testComponents = button_test.components
    for thisComponent in button_test.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "button_test" ---
    button_test.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *move_on* updates
        waitOnFlip = False
        
        # if move_on is starting this frame...
        if move_on.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            move_on.frameNStart = frameN  # exact frame index
            move_on.tStart = t  # local t and not account for scr refresh
            move_on.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(move_on, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'move_on.started')
            # update status
            move_on.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(move_on.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(move_on.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if move_on.status == STARTED and not waitOnFlip:
            theseKeys = move_on.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _move_on_allKeys.extend(theseKeys)
            if len(_move_on_allKeys):
                move_on.keys = _move_on_allKeys[-1].name  # just the last key pressed
                move_on.rt = _move_on_allKeys[-1].rt
                move_on.duration = _move_on_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *button_press* updates
        waitOnFlip = False
        
        # if button_press is starting this frame...
        if button_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            button_press.frameNStart = frameN  # exact frame index
            button_press.tStart = t  # local t and not account for scr refresh
            button_press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_press.started')
            # update status
            button_press.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(button_press.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(button_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if button_press.status == STARTED and not waitOnFlip:
            theseKeys = button_press.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _button_press_allKeys.extend(theseKeys)
            if len(_button_press_allKeys):
                button_press.keys = _button_press_allKeys[-1].name  # just the last key pressed
                button_press.rt = _button_press_allKeys[-1].rt
                button_press.duration = _button_press_allKeys[-1].duration
        
        # *info_press* updates
        
        # if info_press is starting this frame...
        if info_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            info_press.frameNStart = frameN  # exact frame index
            info_press.tStart = t  # local t and not account for scr refresh
            info_press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(info_press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'info_press.started')
            # update status
            info_press.status = STARTED
            info_press.setAutoDraw(True)
        
        # if info_press is active this frame...
        if info_press.status == STARTED:
            # update params
            pass
        
        # *pressed_text* updates
        
        # if pressed_text is starting this frame...
        if pressed_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pressed_text.frameNStart = frameN  # exact frame index
            pressed_text.tStart = t  # local t and not account for scr refresh
            pressed_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pressed_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'pressed_text.started')
            # update status
            pressed_text.status = STARTED
            pressed_text.setAutoDraw(True)
        
        # if pressed_text is active this frame...
        if pressed_text.status == STARTED:
            # update params
            pressed_text.setText(button, log=False)
        # Run 'Each Frame' code from catch_button
        if len(button_press.keys) > 0:
            button = button_press.keys[0]
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            button_test.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in button_test.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "button_test" ---
    for thisComponent in button_test.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for button_test
    button_test.tStop = globalClock.getTime(format='float')
    button_test.tStopRefresh = tThisFlipGlobal
    thisExp.addData('button_test.stopped', button_test.tStop)
    # check responses
    if move_on.keys in ['', [], None]:  # No response was made
        move_on.keys = None
    thisExp.addData('move_on.keys',move_on.keys)
    if move_on.keys != None:  # we had a response
        thisExp.addData('move_on.rt', move_on.rt)
        thisExp.addData('move_on.duration', move_on.duration)
    # check responses
    if button_press.keys in ['', [], None]:  # No response was made
        button_press.keys = None
    thisExp.addData('button_press.keys',button_press.keys)
    if button_press.keys != None:  # we had a response
        thisExp.addData('button_press.rt', button_press.rt)
        thisExp.addData('button_press.duration', button_press.duration)
    thisExp.nextEntry()
    # the Routine "button_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "read_port" ---
    # create an object to store info about Routine read_port
    read_port = data.Routine(
        name='read_port',
        components=[read_port_text, select_pin, display_trigger2, display_trigger3, display_trigger4, display_trigger5, display_trigger6, display_trigger7, display_trigger8, display_trigger9, display_trigger10, display_trigger11, display_trigger12],
    )
    read_port.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for select_pin
    select_pin.keys = []
    select_pin.rt = []
    _select_pin_allKeys = []
    # store start times for read_port
    read_port.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    read_port.tStart = globalClock.getTime(format='float')
    read_port.status = STARTED
    thisExp.addData('read_port.started', read_port.tStart)
    read_port.maxDuration = None
    # keep track of which components have finished
    read_portComponents = read_port.components
    for thisComponent in read_port.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "read_port" ---
    read_port.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from catch_signal
        if port.readPin(10) > 0:
            print("trigger on pin 10")
            print("pin 10:", port.readPin(10))
            counter10 = counter10 + 1
            continueRoutine=False
        
        
        if port.readPin(12) > 0:
            print("trigger on pin 12")
            print("pin 12:", port.readPin(12))
            counter12 = counter12 + 1
        
        # *read_port_text* updates
        
        # if read_port_text is starting this frame...
        if read_port_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            read_port_text.frameNStart = frameN  # exact frame index
            read_port_text.tStart = t  # local t and not account for scr refresh
            read_port_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(read_port_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'read_port_text.started')
            # update status
            read_port_text.status = STARTED
            read_port_text.setAutoDraw(True)
        
        # if read_port_text is active this frame...
        if read_port_text.status == STARTED:
            # update params
            pass
        
        # *select_pin* updates
        waitOnFlip = False
        
        # if select_pin is starting this frame...
        if select_pin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            select_pin.frameNStart = frameN  # exact frame index
            select_pin.tStart = t  # local t and not account for scr refresh
            select_pin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(select_pin, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'select_pin.started')
            # update status
            select_pin.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(select_pin.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(select_pin.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if select_pin.status == STARTED and not waitOnFlip:
            theseKeys = select_pin.getKeys(keyList=['1','2','3','4','5', '6', '7', '8', '9', 'space'], ignoreKeys=["escape"], waitRelease=False)
            _select_pin_allKeys.extend(theseKeys)
            if len(_select_pin_allKeys):
                select_pin.keys = _select_pin_allKeys[-1].name  # just the last key pressed
                select_pin.rt = _select_pin_allKeys[-1].rt
                select_pin.duration = _select_pin_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *display_trigger2* updates
        
        # if display_trigger2 is starting this frame...
        if display_trigger2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger2.frameNStart = frameN  # exact frame index
            display_trigger2.tStart = t  # local t and not account for scr refresh
            display_trigger2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger2.started')
            # update status
            display_trigger2.status = STARTED
            display_trigger2.setAutoDraw(True)
        
        # if display_trigger2 is active this frame...
        if display_trigger2.status == STARTED:
            # update params
            display_trigger2.setText(port.readPin(2) , log=False)
        
        # *display_trigger3* updates
        
        # if display_trigger3 is starting this frame...
        if display_trigger3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger3.frameNStart = frameN  # exact frame index
            display_trigger3.tStart = t  # local t and not account for scr refresh
            display_trigger3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger3.started')
            # update status
            display_trigger3.status = STARTED
            display_trigger3.setAutoDraw(True)
        
        # if display_trigger3 is active this frame...
        if display_trigger3.status == STARTED:
            # update params
            display_trigger3.setText(port.readPin(3) , log=False)
        
        # *display_trigger4* updates
        
        # if display_trigger4 is starting this frame...
        if display_trigger4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger4.frameNStart = frameN  # exact frame index
            display_trigger4.tStart = t  # local t and not account for scr refresh
            display_trigger4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger4.started')
            # update status
            display_trigger4.status = STARTED
            display_trigger4.setAutoDraw(True)
        
        # if display_trigger4 is active this frame...
        if display_trigger4.status == STARTED:
            # update params
            display_trigger4.setText(port.readPin(4) , log=False)
        
        # *display_trigger5* updates
        
        # if display_trigger5 is starting this frame...
        if display_trigger5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger5.frameNStart = frameN  # exact frame index
            display_trigger5.tStart = t  # local t and not account for scr refresh
            display_trigger5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger5.started')
            # update status
            display_trigger5.status = STARTED
            display_trigger5.setAutoDraw(True)
        
        # if display_trigger5 is active this frame...
        if display_trigger5.status == STARTED:
            # update params
            display_trigger5.setText(port.readPin(5) , log=False)
        
        # *display_trigger6* updates
        
        # if display_trigger6 is starting this frame...
        if display_trigger6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger6.frameNStart = frameN  # exact frame index
            display_trigger6.tStart = t  # local t and not account for scr refresh
            display_trigger6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger6.started')
            # update status
            display_trigger6.status = STARTED
            display_trigger6.setAutoDraw(True)
        
        # if display_trigger6 is active this frame...
        if display_trigger6.status == STARTED:
            # update params
            display_trigger6.setText(port.readPin(6) , log=False)
        
        # *display_trigger7* updates
        
        # if display_trigger7 is starting this frame...
        if display_trigger7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger7.frameNStart = frameN  # exact frame index
            display_trigger7.tStart = t  # local t and not account for scr refresh
            display_trigger7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger7.started')
            # update status
            display_trigger7.status = STARTED
            display_trigger7.setAutoDraw(True)
        
        # if display_trigger7 is active this frame...
        if display_trigger7.status == STARTED:
            # update params
            display_trigger7.setText(port.readPin(7) , log=False)
        
        # *display_trigger8* updates
        
        # if display_trigger8 is starting this frame...
        if display_trigger8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger8.frameNStart = frameN  # exact frame index
            display_trigger8.tStart = t  # local t and not account for scr refresh
            display_trigger8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger8.started')
            # update status
            display_trigger8.status = STARTED
            display_trigger8.setAutoDraw(True)
        
        # if display_trigger8 is active this frame...
        if display_trigger8.status == STARTED:
            # update params
            display_trigger8.setText(port.readPin(8) , log=False)
        
        # *display_trigger9* updates
        
        # if display_trigger9 is starting this frame...
        if display_trigger9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger9.frameNStart = frameN  # exact frame index
            display_trigger9.tStart = t  # local t and not account for scr refresh
            display_trigger9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger9.started')
            # update status
            display_trigger9.status = STARTED
            display_trigger9.setAutoDraw(True)
        
        # if display_trigger9 is active this frame...
        if display_trigger9.status == STARTED:
            # update params
            display_trigger9.setText(port.readPin(9) , log=False)
        
        # *display_trigger10* updates
        
        # if display_trigger10 is starting this frame...
        if display_trigger10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger10.frameNStart = frameN  # exact frame index
            display_trigger10.tStart = t  # local t and not account for scr refresh
            display_trigger10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger10.started')
            # update status
            display_trigger10.status = STARTED
            display_trigger10.setAutoDraw(True)
        
        # if display_trigger10 is active this frame...
        if display_trigger10.status == STARTED:
            # update params
            display_trigger10.setText(port.readPin(10) , log=False)
        
        # *display_trigger11* updates
        
        # if display_trigger11 is starting this frame...
        if display_trigger11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger11.frameNStart = frameN  # exact frame index
            display_trigger11.tStart = t  # local t and not account for scr refresh
            display_trigger11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger11.started')
            # update status
            display_trigger11.status = STARTED
            display_trigger11.setAutoDraw(True)
        
        # if display_trigger11 is active this frame...
        if display_trigger11.status == STARTED:
            # update params
            display_trigger11.setText(port.readPin(11) , log=False)
        
        # *display_trigger12* updates
        
        # if display_trigger12 is starting this frame...
        if display_trigger12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            display_trigger12.frameNStart = frameN  # exact frame index
            display_trigger12.tStart = t  # local t and not account for scr refresh
            display_trigger12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(display_trigger12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'display_trigger12.started')
            # update status
            display_trigger12.status = STARTED
            display_trigger12.setAutoDraw(True)
        
        # if display_trigger12 is active this frame...
        if display_trigger12.status == STARTED:
            # update params
            display_trigger12.setText(port.readPin(12) , log=False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            read_port.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in read_port.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "read_port" ---
    for thisComponent in read_port.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for read_port
    read_port.tStop = globalClock.getTime(format='float')
    read_port.tStopRefresh = tThisFlipGlobal
    thisExp.addData('read_port.stopped', read_port.tStop)
    # check responses
    if select_pin.keys in ['', [], None]:  # No response was made
        select_pin.keys = None
    thisExp.addData('select_pin.keys',select_pin.keys)
    if select_pin.keys != None:  # we had a response
        thisExp.addData('select_pin.rt', select_pin.rt)
        thisExp.addData('select_pin.duration', select_pin.duration)
    thisExp.nextEntry()
    # the Routine "read_port" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pin10" ---
    # create an object to store info about Routine pin10
    pin10 = data.Routine(
        name='pin10',
        components=[counter10_text2],
    )
    pin10.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for pin10
    pin10.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pin10.tStart = globalClock.getTime(format='float')
    pin10.status = STARTED
    thisExp.addData('pin10.started', pin10.tStart)
    pin10.maxDuration = None
    # keep track of which components have finished
    pin10Components = pin10.components
    for thisComponent in pin10.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pin10" ---
    pin10.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *counter10_text2* updates
        
        # if counter10_text2 is starting this frame...
        if counter10_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            counter10_text2.frameNStart = frameN  # exact frame index
            counter10_text2.tStart = t  # local t and not account for scr refresh
            counter10_text2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(counter10_text2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'counter10_text2.started')
            # update status
            counter10_text2.status = STARTED
            counter10_text2.setAutoDraw(True)
        
        # if counter10_text2 is active this frame...
        if counter10_text2.status == STARTED:
            # update params
            counter10_text2.setText(counter10, log=False)
        # Run 'Each Frame' code from code
        if port.readPin(10) > 0:
            counter10 = counter10 + 1
         
        if counter10 > 5:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            pin10.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pin10.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pin10" ---
    for thisComponent in pin10.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pin10
    pin10.tStop = globalClock.getTime(format='float')
    pin10.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pin10.stopped', pin10.tStop)
    thisExp.nextEntry()
    # the Routine "pin10" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pin12" ---
    # create an object to store info about Routine pin12
    pin12 = data.Routine(
        name='pin12',
        components=[counter12_text2],
    )
    pin12.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for pin12
    pin12.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pin12.tStart = globalClock.getTime(format='float')
    pin12.status = STARTED
    thisExp.addData('pin12.started', pin12.tStart)
    pin12.maxDuration = None
    # keep track of which components have finished
    pin12Components = pin12.components
    for thisComponent in pin12.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pin12" ---
    pin12.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *counter12_text2* updates
        
        # if counter12_text2 is starting this frame...
        if counter12_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            counter12_text2.frameNStart = frameN  # exact frame index
            counter12_text2.tStart = t  # local t and not account for scr refresh
            counter12_text2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(counter12_text2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'counter12_text2.started')
            # update status
            counter12_text2.status = STARTED
            counter12_text2.setAutoDraw(True)
        
        # if counter12_text2 is active this frame...
        if counter12_text2.status == STARTED:
            # update params
            counter12_text2.setText(counter12, log=False)
        # Run 'Each Frame' code from code_2
        if port.readPin(12) > 0:
            counter12 = counter12 + 1
         
        if counter12 > 5:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            pin12.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pin12.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pin12" ---
    for thisComponent in pin12.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pin12
    pin12.tStop = globalClock.getTime(format='float')
    pin12.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pin12.stopped', pin12.tStop)
    thisExp.nextEntry()
    # the Routine "pin12" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
