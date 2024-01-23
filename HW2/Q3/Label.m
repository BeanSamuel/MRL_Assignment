function [PN] = Label(t,S_n)

PN = -ones(S_n,1);

if t==1
    PN(9) = 1;
    PN(12) = 1;
    return;
elseif t==2
    PN(11) = 1;
    PN(12) = 1;
    return;
elseif t==3
    PN(14) = 1;
    PN(16) = 1;
    return;
elseif t==4
    PN(15) = 1;
    PN(17) = 1;
    return;
elseif t==5
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==6
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==7
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==8
    PN(21) = 1;
    PN(19) = 1;
    return;
elseif t==9
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==10
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==11
    PN(18) = 1;
    return;
elseif t==12
    PN(19) = 1;
    return;
elseif t==13
    PN(18) = 1;
    return;
elseif t==14
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==15
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==16
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==17
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==18
    PN(20) = 1;
    PN(18) = 1;
    return;
elseif t==19
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==20
    PN(17) = 1;
    PN(15) = 1;
    return;
elseif t==21
    PN(13) = 1;
    PN(14) = 1;
    return;
elseif t==22
    PN(12) = 1;
    PN(16) = 1;
    return;
elseif t==23
    PN(9) = 1;
    PN(12) = 1;
    return;
elseif t==24
    PN(9) = 1;
    PN(13) = 1;
    return;
elseif t==25
    PN(9) = 1;
    PN(11) = 1;
    return;
elseif t==26
    PN(9) = 1;
    PN(11) = 1;
    return;
elseif t==27
    PN(8) = 1;
    PN(12) = 1;
    return;
elseif t==28
    PN(8) = 1;
    PN(10) = 1;
    return;
elseif t==29
    PN(9) = 1;
    PN(7) = 1;
    return;
elseif t==30
    PN(4) = 1;
    PN(7) = 1;
    return;
elseif t==31
    PN(4) = 1;
    PN(21) = 1;
    return;
elseif t==32
    PN(18) = 1;
    PN(22) = 1;
    return;
elseif t==33
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==34
    PN(16) = 1;
    PN(17) = 1;
    return;
elseif t==35
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==36
    PN(19) = 1;
    PN(21) = 1;
    return;
elseif t==37
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==38
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==39
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==40
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==41
    PN(16) = 1;
    PN(19) = 1;
    return;
elseif t==42
    PN(19) = 1;
    return;
elseif t==43
    PN(19) = 1;
    PN(22) = 1;
    return;
elseif t==44
    PN(22) = 1;
    PN(24) = 1;
    return;
elseif t==45
    PN(2) = 1;
    PN(25) = 1;
    return;
elseif t==46
    PN(2) = 1;
    PN(23) = 1;
    return;
elseif t==47
    PN(2) = 1;
    PN(3) = 1;
    return;
elseif t==48
    PN(1) = 1;
    PN(3) = 1;
    return;
elseif t==49
    PN(25) = 1;
    PN(27) = 1;
    return;
elseif t==50
    PN(22) = 1;
    PN(23) = 1;
    return;
elseif t==51
    PN(19) = 1;
    PN(22) = 1;
    return;
elseif t==52
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==53
    PN(16) = 1;
    PN(19) = 1;
    return;
elseif t==54
    PN(16) = 1;
    PN(17) = 1;
    return;
elseif t==55
    PN(19) = 1;
    PN(17) = 1;
    return;
elseif t==56
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==57
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==58
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==59
    PN(19) = 1;
    PN(17) = 1;
    return;
elseif t==60
    PN(16) = 1;
    PN(17) = 1;
    return;
elseif t==61
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==62
    PN(15) = 1;
    PN(16) = 1;
    return;
elseif t==63
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==64
    PN(15) = 1;
    return;
elseif t==65
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==66
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==67
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==68
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==69
    PN(18) = 1;
    PN(19) = 1;
    return;
elseif t==70
    PN(16) = 1;
    PN(17) = 1;
    return;
elseif t==71
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==72
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==73
    PN(17) = 1;
    return;
elseif t==74
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==75
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==76
    PN(18) = 1;
    PN(19) = 1;
    return;
elseif t==77
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==78
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==79
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==80
    PN(16) = 1;
    PN(20) = 1;
    return;
elseif t==81
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==82
    PN(13) = 1;
    PN(17) = 1;
    return;
elseif t==83
    PN(14) = 1;
    PN(16) = 1;
    return;
elseif t==84
    PN(13) = 1;
    PN(15) = 1;
    return;
elseif t==85
    PN(13) = 1;
    PN(16) = 1;
    return;
elseif t==86
    PN(14) = 1;
    PN(16) = 1;
    return;
elseif t==87
    PN(13) = 1;
    PN(15) = 1;
    return;
elseif t==88
    PN(9) = 1;
    PN(10) = 1;
    return;
elseif t==89
    PN(7) = 1;
    PN(9) = 1;
    return;
elseif t==90
    PN(7) = 1;
    PN(9) = 1;
    return;
elseif t==91
    PN(8) = 1;
    return;
elseif t==92
    PN(5) = 1;
    PN(7) = 1;
    return;
elseif t==93
    PN(5) = 1;
    PN(24) = 1;
    return;
elseif t==94
    PN(17) = 1;
    PN(18) = 1;
    return;
elseif t==95
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==96
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==97
    PN(13) = 1;
    return;
elseif t==98
    PN(11) = 1;
    PN(12) = 1;
    return;
elseif t==99
    PN(12) = 1;
    PN(15) = 1;
    return;
elseif t==100
    PN(13) = 1;
    PN(15) = 1;
    return;
elseif t==101
    PN(18) = 1;
    PN(20) = 1;
    return;
elseif t==102
    PN(17) = 1;
    PN(13) = 1;
    return;
elseif t==103
    PN(12) = 1;
    PN(14) = 1;
    return;
elseif t==104
    PN(9) = 1;
    PN(12) = 1;
    return;
elseif t==105
    PN(9) = 1;
    return;
elseif t==106
    PN(10) = 1;
    PN(12) = 1;
    return;
elseif t==107
    PN(10) = 1;
    PN(15) = 1;
    return;
elseif t==108
    PN(13) = 1;
    PN(18) = 1;
    return;
elseif t==109
    PN(10) = 1;
    PN(14) = 1;
    return;
elseif t==110
    PN(9) = 1;
    PN(10) = 1;
    return;
elseif t==111
    PN(13) = 1;
    PN(17) = 1;
    return;
elseif t==112
    PN(16) = 1;
    PN(19) = 1;
    return;
elseif t==113
    PN(13) = 1;
    PN(15) = 1;
    return;
elseif t==114
    PN(13) = 1;
    PN(15) = 1;
    return;
elseif t==115
    PN(16) = 1;
    PN(17) = 1;
    return;
elseif t==116
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==117
    PN(18) = 1;
    PN(19) = 1;
    return;
elseif t==118
    PN(17) = 1;
    PN(19) = 1;
    return;
elseif t==119
    PN(16) = 1;
    PN(18) = 1;
    return;
elseif t==120
    PN(18) = 1;
    PN(20) = 1;
    return;

end

end
